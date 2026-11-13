"""Optional full-spectrum signal morphing, without mass extrapolation.

Align each anchor's mean and nominal resolution with the target mass. Linearly
interpolate the aligned unit-area histograms; log-linearly interpolate their
full native-axis yields. Apply the same transformation to every systematic
histogram. Native anchor hypotheses are read unchanged. Fractional bin overlap
assumes constant density inside each native input bin. Sumw2 propagation keeps
the covariance between each partial integral and its anchor normalisation.
It describes finite-anchor statistics, not interpolation modelling uncertainty.
No blinded observations are read.
"""
import bisect
import math
from pathlib import Path

import limit_workflow as base

PREFIX = '__interpolated_spectrum_M-'
METHOD = 'resolution-aligned histogram morph; linear unit-area shape, log-linear full-axis yield; v1'


def requested_files(signals, mass, eras, enabled=False):
    found = next((m for m in signals if abs(m-mass) < 1e-6), None)
    if found is not None:
        files = signals[found]
        if any(era not in files for era in eras):
            raise base.WorkflowError(f'Missing signal era at M={mass:g}')
        return {era: files[era] for era in eras}
    if not enabled:
        raise base.WorkflowError(f'No simulated signal at M={mass:g}; use --interpolate-signal for bracketed intermediate masses.')
    for era in eras:
        base.signal_bracket([m for m, files in signals.items() if era in files], mass)
    return {era: f'{PREFIX}{base.mass_label(mass)}.root' for era in eras}


class HistogramIntegral:
    def __init__(self, hist):
        self.edges = [float(hist.GetXaxis().GetBinLowEdge(i)) for i in range(1, hist.GetNbinsX()+2)]
        self.y = [float(hist.GetBinContent(i)) for i in range(1, hist.GetNbinsX()+1)]
        self.v = [float(hist.GetBinError(i))**2 for i in range(1, hist.GetNbinsX()+1)]
        if any(not math.isfinite(x) for x in (*self.edges, *self.y, *self.v)) or any(a >= b for a,b in zip(self.edges,self.edges[1:])):
            raise base.WorkflowError('Nonfinite or invalid signal histogram')
        self.sy, self.sv = [0.], [0.]
        for y,v in zip(self.y,self.v):
            self.sy.append(self.sy[-1]+y); self.sv.append(self.sv[-1]+v)
        self.total, self.variance = self.sy[-1], self.sv[-1]
        if self.total <= 0:
            raise base.WorkflowError('Signal interpolation requires a positive full-axis integral')

    def partial(self, low, high):
        """Return integral, its variance, covariance with the full integral."""
        low,high=max(low,self.edges[0]),min(high,self.edges[-1])
        if low >= high: return 0.,0.,0.
        first=max(0,bisect.bisect_right(self.edges,low)-1)
        last=min(len(self.y)-1,bisect.bisect_left(self.edges,high)-1)
        if first == last:
            fraction=(high-low)/(self.edges[first+1]-self.edges[first])
            return fraction*self.y[first],fraction*fraction*self.v[first],fraction*self.v[first]
        a=(self.edges[first+1]-low)/(self.edges[first+1]-self.edges[first])
        b=(high-self.edges[last])/(self.edges[last+1]-self.edges[last])
        value=a*self.y[first]+b*self.y[last]+self.sy[last]-self.sy[first+1]
        variance=a*a*self.v[first]+b*b*self.v[last]+self.sv[last]-self.sv[first+1]
        covariance=a*self.v[first]+b*self.v[last]+self.sv[last]-self.sv[first+1]
        return value, max(0.,variance), max(0.,covariance)


def morph_integral(low, high, weight, left, right, mass, left_mass, right_mass,
                   sigma, left_sigma, right_sigma):
    """Analytic independent-anchor Sumw2 propagation, including normalisation."""
    partials=[]
    for hist, anchor, width in ((left,left_mass,left_sigma),(right,right_mass,right_sigma)):
        partials.append(hist.partial(anchor+(low-mass)*width/sigma,
                                     anchor+(high-mass)*width/sigma))
    qleft, qright = partials[0][0]/left.total, partials[1][0]/right.total
    q=(1-weight)*qleft+weight*qright
    total=math.exp((1-weight)*math.log(left.total)+weight*math.log(right.total))
    variance=0.
    for hist, (integral,var,cov), fraction, qi in zip((left,right),partials,(1-weight,weight),(qleft,qright)):
        c=q-qi
        variance+=(total*fraction/hist.total)**2*(var+2*c*cov+c*c*hist.variance)
    value=total*q
    if not math.isfinite(value) or not math.isfinite(variance):
        raise base.WorkflowError('Nonfinite interpolated signal yield/error')
    return base.YieldResult(value,math.sqrt(max(0.,variance)),1,0)


class SpectrumSignalReader:
    def __init__(self, reader, args, signals, mass, era):
        self.reader,self.args,self.mass,self.era=reader,args,mass,era
        self.signals={float(m):files for m,files in signals.items()}
        self.low,self.high=base.signal_bracket([m for m,files in self.signals.items() if era in files],mass)
        self.cache={}

    def __getattr__(self,name): return getattr(self.reader,name)

    def _hist(self, source, anchor, path):
        filename=base.file_for_process(self.args,self.era,'sig',self.signals[anchor][self.era],source)
        key=(filename,path)
        if key not in self.cache:
            hist=self.reader._file(filename).Get(path)
            if not hist or not hasattr(hist,'GetNbinsX'):
                raise base.WorkflowError(f'Missing interpolation anchor histogram {filename}:{path}')
            self.cache[key]=HistogramIntegral(hist)
        return self.cache[key]

    def preflight(self):
        for source,path in base.signal_template_requests(self.args,self.era):
            for anchor in (self.low,self.high): self._hist(source,anchor,path)

    def integral(self,filename,path,low,high):
        if not Path(filename).name.startswith(PREFIX):
            return self.reader.integral(filename,path,low,high)
        if self.low == self.high:
            raise base.WorkflowError('Native signals must bypass morphing')
        parent=Path(filename).parent
        source='experimental' if 'RunSyst' in parent.parts else 'theory' if 'RunXSecSyst' in parent.parts else 'nominal'
        left,right=self._hist(source,self.low,path),self._hist(source,self.high,path)
        sigma=lambda mass: mass*base.relative_resolution(self.args,self.era,mass)
        return morph_integral(low,high,(self.mass-self.low)/(self.high-self.low),left,right,
                              self.mass,self.low,self.high,sigma(self.mass),sigma(self.low),sigma(self.high))


def preflight(reader,args,signals,masses,eras):
    records=[]
    for mass in masses:
        files=requested_files(signals,mass,eras,True)
        for era in eras:
            if not Path(files[era]).name.startswith(PREFIX): continue
            wrapped=SpectrumSignalReader(reader,args,signals,mass,era)
            wrapped.preflight()
            records.append(dict(mass=mass,era=era,anchors=[wrapped.low,wrapped.high],method=METHOD,
                                resolution=args.resolution_coefficients[era]))
    return records
