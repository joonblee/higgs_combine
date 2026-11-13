"""Counting-yield interpolation shared with limit_workflow.py.

Native simulated masses keep their original histogram integrals. At an
intermediate mass, the SR count and all its systematic variations use
base.SignalInterpolator: log-linear yields in each anchor's own +/-n-sigma
window, with the same independent-anchor Sumw2 propagation. No parametric
signal PDF or whole-histogram shape morph is constructed.

Sideband/spectrum channels still need counts beyond the original SR-only
workflow. Map the target SR and its two sidebands onto the corresponding
anchor regions, keeping 11 and 80 GeV fixed. Map each channel boundary
piecewise linearly and snap it to the nearest native anchor edge. Shared
boundaries stay shared, so native anchor bins never overlap between channels.
Interpolate those region counts with the same log-linear counting helper.
Only nonpositive tail counts use signed linear interpolation; a nonpositive
SR anchor remains an error, exactly as in the original counting workflow.
This extension retains signal leakage in sidebands. No mass extrapolation,
new interpolation nuisance, background change or observation read is made.
"""
import bisect
import math
from pathlib import Path

import limit_workflow as base

PREFIX = '__interpolated_spectrum_M-'
VERSION = 'counting_v2'
METHOD = ('limit_workflow.SignalInterpolator SR counts; corresponding whole-bin '
          'sideband/spectrum counts; log-linear positive yields; '+VERSION)
DOMAIN = (11.,80.)


def requested_files(signals, mass, eras, enabled=False):
    found = next((m for m in signals if abs(float(m)-mass) < 1e-6), None)
    if found is not None:
        files = signals[found]
        if any(era not in files for era in eras):
            raise base.WorkflowError(f'Missing signal era at M={mass:g}')
        return {era: files[era] for era in eras}
    if not enabled:
        raise base.WorkflowError(f'No simulated signal at M={mass:g}; use --interpolate-signal for bracketed intermediate masses.')
    for era in eras:
        base.signal_bracket([float(m) for m,files in signals.items() if era in files],mass)
    return {era:f'{PREFIX}{base.mass_label(mass)}.root' for era in eras}


def interpolate_region(mass, left_mass, left, right_mass, right, is_sr=False):
    """Use the legacy counting interpolation; allow empty/signed tails only."""
    if is_sr or (left.value > 0. and right.value > 0.):
        return base.interpolate_signal_yield(mass,left_mass,left,right_mass,right)
    for result in (left,right):
        if not all(math.isfinite(v) for v in (result.value,result.error)) or result.error < 0.:
            raise base.WorkflowError('Invalid signal tail count/error')
    t=(mass-left_mass)/(right_mass-left_mass)
    if not 0. < t < 1.:
        raise base.WorkflowError('Signal tail interpolation requires bracketed masses')
    return base.YieldResult((1.-t)*left.value+t*right.value,
                           math.hypot((1.-t)*left.error,t*right.error),1,0)


def nearest_edge(edges, value):
    i=bisect.bisect_left(edges,value)
    candidates=edges[max(0,i-1):min(len(edges),i+1)]
    return min(candidates,key=lambda x:(abs(x-value),x))


def map_edge(value, target_window, anchor_window, edges):
    """One deterministic shared-edge map; never independently expand each bin."""
    target=(DOMAIN[0],*target_window,DOMAIN[1])
    anchor=(DOMAIN[0],*anchor_window,DOMAIN[1])
    if value < DOMAIN[0]-1e-7 or value > DOMAIN[1]+1e-7:
        raise base.WorkflowError('Signal count requested outside 11--80 GeV')
    # A clipped target SR may coincide with a domain boundary. Its counting
    # edge takes precedence; an absent sideband contributes no extra tail.
    for x,y in [*zip(target[1:3],anchor[1:3]),(target[0],anchor[0]),(target[3],anchor[3])]:
        if abs(value-x) < 1e-8: return y
    index=min(2,max(0,bisect.bisect_right(target,value)-1))
    a,b=target[index:index+2];c,d=anchor[index:index+2]
    if b <= a: raise base.WorkflowError('Degenerate signal-region mapping')
    return nearest_edge(edges,c+(value-a)*(d-c)/(b-a))


class NativeCountingReader:
    """Snap analytic SR windows exactly as RootReader before blinded reading."""
    def __init__(self,owner): self.owner=owner
    def integral(self,filename,path,low,high):
        handle=self.owner.reader._file(filename)
        hist=handle.Get(path) if handle else None
        if not hist or not hasattr(hist,'GetNbinsX'): return None
        left,right=base.whole_bin_window(hist,low,high)
        return self.owner.reader.integral(filename,path,left,right)


class SpectrumSignalReader:
    """Route signal counts only; all backgrounds delegate to the original reader."""
    def __init__(self,reader,args,signals,mass,era,preflight_cache=None):
        self.reader,self.args,self.mass,self.era=reader,args,mass,era
        self.signals={float(m):files for m,files in signals.items()}
        # This cache belongs to one preflight invocation only. Anchor yields,
        # geometry and input checks do not depend on the intermediate mass.
        self.preflight_cache=preflight_cache if preflight_cache is not None else {}
        references=self.preflight_cache.setdefault('references',{})
        if era not in references:
            references[era]=base.SignalInterpolator(NativeCountingReader(self),args,self.signals,[era])
        self.reference=references[era]
        self.low,self.high=self.reference.bracket(era,mass)
        self.histograms={};self.region_cache={}
        self.edges={};self.anchor_windows={}
        geometry_cache=self.preflight_cache.setdefault('geometry',{})
        for anchor in (self.low,self.high):
            key=(era,anchor)
            if key not in geometry_cache:
                hist=self._hist('nominal',anchor,base.hist_path(args.region))
                axis=[float(hist.GetXaxis().GetBinLowEdge(i)) for i in range(1,hist.GetNbinsX()+2)]
                geometry_cache[key]=(axis,base.whole_bin_window(hist,*base.mass_window(args,era,anchor)))
            self.edges[anchor],self.anchor_windows[anchor]=geometry_cache[key]
        geometry=getattr(args,'signal_counting_windows',{}).get(base.mass_label(mass),{}).get(era,[])
        sr=next((row for row in geometry if row['region']=='SR'),None)
        self.target_window=((sr['low'],sr['high']) if sr else base.whole_bin_window(
            self._hist('nominal',self.low,base.hist_path(args.region)),*base.mass_window(args,era,mass)))
        self.redirects={base.file_for_process(args,era,'sig',f'{PREFIX}{base.mass_label(mass)}.root',source):source
                        for source in ('nominal','experimental','theory')}

    def __getattr__(self,name): return getattr(self.reader,name)

    def _hist(self,source,anchor,path):
        filename=base.file_for_process(self.args,self.era,'sig',self.signals[anchor][self.era],source)
        key=(filename,path)
        if key not in self.histograms:
            handle=self.reader._file(filename)
            hist=handle.Get(path) if handle else None
            if not hist or not hasattr(hist,'GetNbinsX'):
                raise base.WorkflowError(f'Missing counting anchor histogram {filename}:{path}')
            self.histograms[key]=hist
        return self.histograms[key]

    def preflight(self):
        validated=self.preflight_cache.setdefault('validated',set())
        for source,path in base.signal_template_requests(self.args,self.era):
            for anchor in (self.low,self.high):
                key=(self.era,anchor,source,path)
                if key in validated: continue
                hist=self._hist(source,anchor,path)
                axis=[float(hist.GetXaxis().GetBinLowEdge(i)) for i in range(1,hist.GetNbinsX()+2)]
                if len(axis)!=len(self.edges[anchor]) or any(abs(a-b)>1e-8 for a,b in zip(axis,self.edges[anchor])):
                    raise base.WorkflowError('Signal systematic anchor has incompatible native binning')
                self.reference.native(self.era,anchor,source,path)
                validated.add(key)
        return self.reference.evaluate(self.era,self.mass,'nominal',base.hist_path(self.args.region))

    def integral(self,filename,path,low,high):
        if not Path(filename).name.startswith(PREFIX):
            return self.reader.integral(filename,path,low,high)
        try: source=self.redirects[filename]
        except KeyError as exc: raise base.WorkflowError('Unknown counting-interpolation signal path '+filename) from exc
        if all(abs(a-b)<1e-8 for a,b in zip((low,high),self.target_window)):
            return self.reference.evaluate(self.era,self.mass,source,path)
        results=[]
        for anchor in (self.low,self.high):
            bounds=tuple(map_edge(x,self.target_window,self.anchor_windows[anchor],self.edges[anchor]) for x in (low,high))
            key=(anchor,source,path,*bounds)
            if key not in self.region_cache:
                if bounds[0]>=bounds[1]: result=base.YieldResult(0.,0.,1,0)
                else:
                    actual=base.file_for_process(self.args,self.era,'sig',self.signals[anchor][self.era],source)
                    result=self.reader.integral(actual,path,*bounds)
                    if result is None: raise base.WorkflowError('Missing interpolated signal channel count')
                self.region_cache[key]=result
            results.append(self.region_cache[key])
        return interpolate_region(self.mass,self.low,results[0],self.high,results[1])


def preflight(reader,args,signals,masses,eras):
    records=[];cache={}
    print(f'[SIGNAL] Interpolation preflight: {len(masses)} masses, {len(eras)} eras; checking each required anchor once',flush=True)
    for index,mass in enumerate(masses,1):
        files=requested_files(signals,mass,eras,True)
        for era in eras:
            if not Path(files[era]).name.startswith(PREFIX): continue
            wrapped=SpectrumSignalReader(reader,args,signals,mass,era,preflight_cache=cache)
            nominal=wrapped.preflight()
            records.append(dict(mass=mass,era=era,anchors=[wrapped.low,wrapped.high],method=METHOD,
                resolution=args.resolution_coefficients[era],window=list(wrapped.target_window),
                raw_sr_yield=nominal.value,raw_sr_mc_error=nominal.error,
                sidebands='corresponding whole-bin regions; no fractional native-bin splitting',
                nonpositive_tail_interpolation='signed linear; SR requires positive anchors'))
        if index==1 or index%20==0 or index==len(masses):
            print(f'[SIGNAL] Preflight {index}/{len(masses)} masses; {len(records)} interpolated mass/era yields',flush=True)
    return records
