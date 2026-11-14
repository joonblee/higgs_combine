"""PyROOT helpers for the NIsoMuon limit plot and its optional ratio panel."""
import ROOT as R
import math
from array import array
import json
import ctypes


def SetTDRStyle():
    """Sets the PubComm recommended style

    Just a copy of <http://ghm.web.cern.ch/ghm/plots/MacroExample/tdrstyle.C>
    @sa ModTDRStyle() to use this style with some additional customisation.
    """
    # For the canvas:
    R.gStyle.SetCanvasBorderMode(0)
    R.gStyle.SetCanvasColor(R.kWhite)
    R.gStyle.SetCanvasDefH(600)  # Height of canvas
    R.gStyle.SetCanvasDefW(600)  # Width of canvas
    R.gStyle.SetCanvasDefX(0)  # POsition on screen
    R.gStyle.SetCanvasDefY(0)

    # For the Pad:
    R.gStyle.SetPadBorderMode(0)
    R.gStyle.SetPadColor(R.kWhite)
    R.gStyle.SetPadGridX(False)
    R.gStyle.SetPadGridY(False)
    R.gStyle.SetGridColor(0)
    R.gStyle.SetGridStyle(3)
    R.gStyle.SetGridWidth(1)

    # For the frame:
    R.gStyle.SetFrameBorderMode(0)
    R.gStyle.SetFrameBorderSize(1)
    R.gStyle.SetFrameFillColor(0)
    R.gStyle.SetFrameFillStyle(0)
    R.gStyle.SetFrameLineColor(1)
    R.gStyle.SetFrameLineStyle(1)
    R.gStyle.SetFrameLineWidth(1)

    # For the histo:
    R.gStyle.SetHistLineColor(1)
    R.gStyle.SetHistLineStyle(0)
    R.gStyle.SetHistLineWidth(1)

    R.gStyle.SetEndErrorSize(2)

    R.gStyle.SetMarkerStyle(20)

    # For the fit/function:
    R.gStyle.SetOptFit(1)
    R.gStyle.SetFitFormat("5.4g")
    R.gStyle.SetFuncColor(2)
    R.gStyle.SetFuncStyle(1)
    R.gStyle.SetFuncWidth(1)

    # For the date:
    R.gStyle.SetOptDate(0)

    # For the statistics box:
    R.gStyle.SetOptFile(0)
    R.gStyle.SetOptStat(0)
    # To display the mean and RMS:   SetOptStat('mr')
    R.gStyle.SetStatColor(R.kWhite)
    R.gStyle.SetStatFont(42)
    R.gStyle.SetStatFontSize(0.025)
    R.gStyle.SetStatTextColor(1)
    R.gStyle.SetStatFormat("6.4g")
    R.gStyle.SetStatBorderSize(1)
    R.gStyle.SetStatH(0.1)
    R.gStyle.SetStatW(0.15)

    # Margins:
    R.gStyle.SetPadTopMargin(0.05)
    R.gStyle.SetPadBottomMargin(0.13)
    R.gStyle.SetPadLeftMargin(0.16)
    R.gStyle.SetPadRightMargin(0.02)

    # For the Global title:
    R.gStyle.SetOptTitle(0)
    R.gStyle.SetTitleFont(42)
    R.gStyle.SetTitleColor(1)
    R.gStyle.SetTitleTextColor(1)
    R.gStyle.SetTitleFillColor(10)
    R.gStyle.SetTitleFontSize(0.05)

    # For the axis titles:
    R.gStyle.SetTitleColor(1, "XYZ")
    R.gStyle.SetTitleFont(42, "XYZ")
    R.gStyle.SetTitleSize(0.06, "XYZ")
    # Another way to set the size?
    R.gStyle.SetTitleXOffset(0.9)
    R.gStyle.SetTitleYOffset(1.25)

    # For the axis labels:

    R.gStyle.SetLabelColor(1, "XYZ")
    R.gStyle.SetLabelFont(42, "XYZ")
    R.gStyle.SetLabelOffset(0.007, "XYZ")
    R.gStyle.SetLabelSize(0.05, "XYZ")

    # For the axis:

    R.gStyle.SetAxisColor(1, "XYZ")
    R.gStyle.SetStripDecimals(True)
    R.gStyle.SetTickLength(0.03, "XYZ")
    R.gStyle.SetNdivisions(510, "XYZ")
    R.gStyle.SetPadTickX(1)
    R.gStyle.SetPadTickY(1)

    # Change for log plots:
    R.gStyle.SetOptLogx(0)
    R.gStyle.SetOptLogy(0)
    R.gStyle.SetOptLogz(0)

    # Postscript options:
    R.gStyle.SetPaperSize(20.0, 20.0)


    R.gStyle.SetHatchesLineWidth(5)
    R.gStyle.SetHatchesSpacing(0.05)


def ModTDRStyle(width=600, height=600, t=0.06, b=0.12, l=0.16, r=0.04):
    """Modified version of the tdrStyle

    Args:
        width (int): Canvas width in pixels
        height (int): Canvas height in pixels
        t (float): Pad top margin [0-1]
        b (float): Pad bottom margin [0-1]
        l (float): Pad left margin [0-1]
        r (float): Pad right margin [0-1]
    """
    SetTDRStyle()

    # Set the default canvas width and height in pixels
    R.gStyle.SetCanvasDefW(width)
    R.gStyle.SetCanvasDefH(height)

    # Set the default margins. These are given as fractions of the pad height
    # for `Top` and `Bottom` and the pad width for `Left` and `Right`. But we
    # want to specify all of these as fractions of the shortest length.
    def_w = float(R.gStyle.GetCanvasDefW())
    def_h = float(R.gStyle.GetCanvasDefH())

    scale_h = (def_w / def_h) if (def_h > def_w) else 1.0
    scale_w = (def_h / def_w) if (def_w > def_h) else 1.0

    def_min = def_h if (def_h < def_w) else def_w

    R.gStyle.SetPadTopMargin(t * scale_h)
    # default 0.05
    R.gStyle.SetPadBottomMargin(b * scale_h)
    # default 0.13
    R.gStyle.SetPadLeftMargin(l * scale_w)
    # default 0.16
    R.gStyle.SetPadRightMargin(r * scale_w)
    # default 0.02
    # But note the new CMS style sets these:
    # 0.08, 0.12, 0.12, 0.04

    # Set number of axis tick divisions
    R.gStyle.SetNdivisions(506, "XYZ")  # default 510

    # Some marker properties not set in the default tdr style
    R.gStyle.SetMarkerColor(R.kBlack)
    R.gStyle.SetMarkerSize(1.0)

    R.gStyle.SetLabelOffset(0.007, "YZ")
    # This is an adhoc adjustment to scale the x-axis label
    # offset when we stretch plot vertically
    # Will also need to increase if first x-axis label has more than one digit
    R.gStyle.SetLabelOffset(0.005 * (3.0 - 2.0 / scale_h), "X")

    # In this next part we do a slightly involved calculation to set the axis
    # title offsets, depending on the values of the TPad dimensions and
    # margins. This is to try and ensure that regardless of how these pad
    # values are set, the axis titles will be located towards the edges of the
    # canvas and not get pushed off the edge - which can often happen if a
    # fixed value is used.
    title_size = 0.05
    title_px = title_size * def_min
    label_size = 0.04
    R.gStyle.SetTitleSize(title_size, "XYZ")
    R.gStyle.SetLabelSize(label_size, "XYZ")

    R.gStyle.SetTitleXOffset(0.5 * scale_h * (1.2 * (def_h * b * scale_h - 0.6 * title_px)) / title_px)
    R.gStyle.SetTitleYOffset(0.5 * scale_w * (1.2 * (def_w * l * scale_w - 0.6 * title_px)) / title_px)

    # Only draw ticks where we have an axis
    R.gStyle.SetPadTickX(0)
    R.gStyle.SetPadTickY(0)
    R.gStyle.SetTickLength(0.02, "XYZ")

    R.gStyle.SetLegendBorderSize(0)
    R.gStyle.SetLegendFont(42)
    R.gStyle.SetLegendFillColor(0)
    R.gStyle.SetFillColor(0)

    R.gROOT.ForceStyle()


def Set(obj, **kwargs):
    for key, value in kwargs.items():
        if value is None:
            getattr(obj, "Set" + key)()
        elif isinstance(value, (list, tuple)):
            getattr(obj, "Set" + key)(*value)
        else:
            getattr(obj, "Set" + key)(value)


def OnePad():
    pad = R.TPad("pad", "pad", 0.0, 0.0, 1.0, 1.0)
    pad.Draw()
    pad.cd()
    result = [pad]
    return result


def TwoPadSplit(split_point, gap_low, gap_high):
    upper = R.TPad("upper", "upper", 0.0, 0.0, 1.0, 1.0)
    upper.SetBottomMargin(split_point + gap_high)
    upper.SetFillStyle(4000)
    upper.Draw()
    lower = R.TPad("lower", "lower", 0.0, 0.0, 1.0, 1.0)
    lower.SetTopMargin(1 - split_point + gap_low)
    lower.SetFillStyle(4000)
    lower.Draw()
    upper.cd()
    result = [upper, lower]
    return result


def SetupTwoPadSplitAsRatio(pads, upper, lower, y_title, y_centered, y_min, y_max):
    if lower.GetXaxis().GetTitle() == "":
        lower.GetXaxis().SetTitle(upper.GetXaxis().GetTitle())
    upper.GetXaxis().SetTitle("")
    upper.GetXaxis().SetLabelSize(0)
    upper_h = 1.0 - pads[0].GetTopMargin() - pads[0].GetBottomMargin()
    lower_h = 1.0 - pads[1].GetTopMargin() - pads[1].GetBottomMargin()
    lower.GetYaxis().SetTickLength(R.gStyle.GetTickLength() * upper_h / lower_h)
    pads[1].SetTickx(1)
    pads[1].SetTicky(1)
    lower.GetYaxis().SetTitle(y_title)
    lower.GetYaxis().CenterTitle(y_centered)
    if y_max > y_min:
        lower.SetMinimum(y_min)
        lower.SetMaximum(y_max)


def CreateAxisHist(src, at_limits=True):
    backup = R.gPad
    tmp = R.TCanvas()
    tmp.cd()
    src.Draw("AP")
    result = src.GetHistogram().Clone("tmp")
    if at_limits:
        x = ctypes.c_double(0.0)
        y = ctypes.c_double(0.0)
        src.GetPoint(0, x, y)
        min = float(x.value)
        max = float(x.value)
        for i in range(1, src.GetN()):
            src.GetPoint(i, x, y)
            if float(x.value) < min:
                min = float(x.value)
            if float(x.value) > max:
                max = float(x.value)
        result.GetXaxis().SetLimits(min, max)
    R.gPad = backup
    return result


def CreateAxisHists(n, src, at_limits):
    res = []
    h = CreateAxisHist(src, at_limits)
    for i in range(n):
        res.append(h.Clone("tmp%i" % i))
    return res


def GetAxisHist(pad):
    pad_obs = pad.GetListOfPrimitives()
    if pad_obs is None:
        return None
    obj = None
    for obj in pad_obs:
        if obj.InheritsFrom(R.TH1.Class()):
            return obj
        if obj.InheritsFrom(R.TMultiGraph.Class()):
            return obj.GetHistogram()
        if obj.InheritsFrom(R.TGraph.Class()):
            return obj.GetHistogram()
        if obj.InheritsFrom(R.THStack.Class()):
            return obj.GetHistogram()
    return None


def LimitTGraphFromJSON(js, label):
    xvals = []
    yvals = []
    for key in js:
        xvals.append(float(key))
        yvals.append(js[key][label])
    graph = R.TGraph(len(xvals), array("d", xvals), array("d", yvals))
    graph.Sort()
    return graph


def LimitTGraphFromJSONFile(jsfile, label):
    with open(jsfile) as jsonfile:
        js = json.load(jsonfile)
    return LimitTGraphFromJSON(js, label)


def LimitBandTGraphFromJSON(js, central, lo, hi):
    xvals = []
    yvals = []
    yvals_lo = []
    yvals_hi = []
    for key in js:
        xvals.append(float(key))
        yvals.append(js[key][central])
        yvals_lo.append(js[key][central] - js[key][lo])
        yvals_hi.append(js[key][hi] - js[key][central])
    graph = R.TGraphAsymmErrors(len(xvals), array("d", xvals), array("d", yvals), array("d", [0]), array("d", [0]), array("d", yvals_lo), array("d", yvals_hi))
    graph.Sort()
    return graph


def StandardLimitsFromJSONFile(json_file, draw=["obs", "exp0", "exp1", "exp2"]):
    graphs = {}
    data = {}
    with open(json_file) as jsonfile:
        data = json.load(jsonfile)
    if "obs" in draw:
        graphs["obs"] = LimitTGraphFromJSON(data, "obs")
    if "exp0" in draw or "exp" in draw:
        graphs["exp0"] = LimitTGraphFromJSON(data, "exp0")
    if "exp1" in draw or "exp" in draw:
        graphs["exp1"] = LimitBandTGraphFromJSON(data, "exp0", "exp-1", "exp+1")
    if "exp2" in draw or "exp" in draw:
        graphs["exp2"] = LimitBandTGraphFromJSON(data, "exp0", "exp-2", "exp+2")
    return graphs


def GraphDivide(num, den):
    res = num.Clone()
    for i in range(num.GetN()):
        res.GetY()[i] = res.GetY()[i] / den.Eval(res.GetX()[i])
    if isinstance(res, R.TGraphAsymmErrors):
        for i in range(num.GetN()):
            res.GetEYhigh()[i] = res.GetEYhigh()[i] / den.Eval(res.GetX()[i])
            res.GetEYlow()[i] = res.GetEYlow()[i] / den.Eval(res.GetX()[i])

    return res


def FixBothRanges(pad, fix_y_lo, frac_lo, fix_y_hi, frac_hi):
    """Adjusts y-axis range such that a lower and a higher value are located a
    fixed fraction of the frame height away from a new minimum and maximum
    respectively.

    This function is useful in conjunction with GetPadYMax which returns the
    maximum or minimum y value of all histograms and graphs drawn on the pad.

    In the example below, the minimum and maximum values found via this function
    are used as the `fix_y_lo` and `fix_y_hi` arguments, and the spacing fractions
    as 0.15 and 0.30 respectively.

    @code
    FixBothRanges(pad, GetPadYMin(pad), 0.15, GetPadYMax(pad), 0.30)
    @endcode

    ![](figures/FixBothRanges.png)

    Args:
        pad (TPad): A TPad on which histograms and graphs have already been drawn
        fix_y_lo (float): The y value which will end up a fraction `frac_lo` above
                          the new axis minimum.
        frac_lo (float): A fraction of the y-axis height
        fix_y_hi (float): The y value which will end up a fraction `frac_hi` below
                         from the new axis maximum.
        frac_hi (float): A fraction of the y-axis height
    """
    hobj = GetAxisHist(pad)
    ymin = fix_y_lo
    ymax = fix_y_hi
    if R.gPad.GetLogy():
        if ymin == 0.0:
            print("Cannot adjust log-scale y-axis range if the minimum is zero!")
            return
        ymin = math.log10(ymin)
        ymax = math.log10(ymax)
    fl = frac_lo
    fh = frac_hi

    ymaxn = (1.0 / (1.0 - (fh * fl / ((1.0 - fl) * (1.0 - fh))))) * (1.0 / (1.0 - fh)) * (ymax - fh * ymin)
    yminn = (ymin - fl * ymaxn) / (1.0 - fl)
    if R.gPad.GetLogy():
        yminn = math.pow(10, yminn)
        ymaxn = math.pow(10, ymaxn)
    hobj.SetMinimum(yminn)
    hobj.SetMaximum(ymaxn)


def GetPadYMaxInRange(pad, x_min, x_max, do_min=False):
    pad_obs = pad.GetListOfPrimitives()
    if pad_obs is None:
        return 0.0
    h_max = -99999.0
    h_min = +99999.0
    for obj in pad_obs:
        if obj.InheritsFrom(R.TH1.Class()):
            hobj = obj
            for j in range(1, hobj.GetNbinsX() + 1):
                if hobj.GetBinLowEdge(j) + hobj.GetBinWidth(j) < x_min or hobj.GetBinLowEdge(j) > x_max:
                    continue
                if hobj.GetBinContent(j) + hobj.GetBinError(j) > h_max:
                    h_max = hobj.GetBinContent(j) + hobj.GetBinError(j)
                if (hobj.GetBinContent(j) - hobj.GetBinError(j) < h_min) and not do_min:
                    # If we're looking for the minimum don't count TH1s
                    # because we probably only care about graphs
                    h_min = hobj.GetBinContent(j) - hobj.GetBinError(j)
        elif obj.InheritsFrom(R.TGraphAsymmErrors.Class()):
            gobj = obj
            n = gobj.GetN()
            for k in range(0, n):
                x = gobj.GetX()[k]
                y = gobj.GetY()[k]
                if x < x_min or x > x_max:
                    continue
                if (y + gobj.GetEYhigh()[k]) > h_max:
                    h_max = y + gobj.GetEYhigh()[k]
                if (y - gobj.GetEYlow()[k]) < h_min:
                    h_min = y - gobj.GetEYlow()[k]
        elif obj.InheritsFrom(R.TGraphErrors.Class()):
            gobj = obj
            n = gobj.GetN()
            for k in range(0, n):
                x = gobj.GetX()[k]
                y = gobj.GetY()[k]
                if x < x_min or x > x_max:
                    continue
                if (y + gobj.GetEY()[k]) > h_max:
                    h_max = y + gobj.GetEY()[k]
                if (y - gobj.GetEY()[k]) < h_min:
                    h_min = y - gobj.GetEY()[k]
        elif obj.InheritsFrom(R.TGraph.Class()):
            gobj = obj
            n = gobj.GetN()
            for k in range(0, n):
                x = gobj.GetX()[k]
                y = gobj.GetY()[k]
                if x < x_min or x > x_max:
                    continue
                if y > h_max:
                    h_max = y
                if y < h_min:
                    h_min = y
    return h_max if do_min is False else h_min


def GetPadYMax(pad, do_min=False):
    pad_obs = pad.GetListOfPrimitives()
    if pad_obs is None:
        return 0.0
    xmin = GetAxisHist(pad).GetXaxis().GetXmin()
    xmax = GetAxisHist(pad).GetXaxis().GetXmax()
    return GetPadYMaxInRange(pad, xmin, xmax, do_min)


def GetPadYMin(pad):
    return GetPadYMax(pad, True)


def DrawCMSLogo(pad, cmsText, extraText, iPosX, relPosX, relPosY, relExtraDY, extraText2="", cmsTextSize=0.8):
    """Draw the CMS label in the selected pad."""
    pad.cd()
    cmsTextFont = 62  # default is helvetic-bold

    writeExtraText = len(extraText) > 0
    writeExtraText2 = len(extraText2) > 0
    extraTextFont = 52

    # text sizes and text offsets with respect to the top frame
    # in unit of the top margin size
    lumiTextOffset = 0.2
    # cmsTextSize = 0.8
    # float cmsTextOffset    = 0.1;  // only used in outOfFrame version

    # ratio of 'CMS' and extra text size
    extraOverCmsTextSize = 0.76

    outOfFrame = False
    if iPosX / 10 == 0:
        outOfFrame = True

    alignY_ = 3
    alignX_ = 2
    if iPosX / 10 == 0:
        alignX_ = 1
    if iPosX == 0:
        alignX_ = 1
    if iPosX == 0:
        alignY_ = 1
    if iPosX / 10 == 1:
        alignX_ = 1
    if iPosX / 10 == 2:
        alignX_ = 2
    if iPosX / 10 == 3:
        alignX_ = 3
    # if (iPosX == 0): relPosX = 0.14
    align_ = 10 * alignX_ + alignY_

    l = pad.GetLeftMargin()
    t = pad.GetTopMargin()
    r = pad.GetRightMargin()
    b = pad.GetBottomMargin()

    latex = R.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(R.kBlack)

    extraTextSize = extraOverCmsTextSize * cmsTextSize
    pad_ratio = (float(pad.GetWh()) * pad.GetAbsHNDC()) / (float(pad.GetWw()) * pad.GetAbsWNDC())
    if pad_ratio < 1.0:
        pad_ratio = 1.0

    if outOfFrame:
        latex.SetTextFont(cmsTextFont)
        latex.SetTextAlign(11)
        latex.SetTextSize(cmsTextSize * t * (1.0 / pad_ratio))
        latex.DrawLatex(l, 1 - t + lumiTextOffset * t, cmsText)

    posX_ = 0
    if iPosX % 10 <= 1:
        posX_ = l + relPosX * (1 - l - r)
    elif iPosX % 10 == 2:
        posX_ = l + 0.5 * (1 - l - r)
    elif iPosX % 10 == 3:
        posX_ = 1 - r - relPosX * (1 - l - r)

    posY_ = 1 - t - relPosY * (1 - t - b)
    if not outOfFrame:
        latex.SetTextFont(cmsTextFont)
        latex.SetTextSize(cmsTextSize * t * pad_ratio)
        latex.SetTextAlign(align_)
        latex.DrawLatex(posX_, posY_, cmsText)
        if writeExtraText:
            latex.SetTextFont(extraTextFont)
            latex.SetTextAlign(align_)
            latex.SetTextSize(extraTextSize * t * pad_ratio)
            latex.DrawLatex(posX_, posY_ - relExtraDY * cmsTextSize * t, extraText)
            if writeExtraText2:
                latex.DrawLatex(posX_, posY_ - 1.8 * relExtraDY * cmsTextSize * t, extraText2)
    elif writeExtraText:
        if iPosX == 0:
            posX_ = l + relPosX * (1 - l - r)
            posY_ = 1 - t + lumiTextOffset * t
        latex.SetTextFont(extraTextFont)
        latex.SetTextSize(extraTextSize * t * (1.0 / pad_ratio))
        latex.SetTextAlign(align_)
        latex.DrawLatex(posX_, posY_, extraText)


def PositionedLegend(width, height, pos, offset, horizontaloffset=None):
    o = offset
    ho = horizontaloffset
    if not ho:
        ho = o
    w = width
    h = height
    l = R.gPad.GetLeftMargin()
    t = R.gPad.GetTopMargin()
    b = R.gPad.GetBottomMargin()
    r = R.gPad.GetRightMargin()
    if pos == 1:
        return R.TLegend(l + ho, 1 - t - o - h, l + ho + w, 1 - t - o, "", "NBNDC")
    if pos == 2:
        c = l + 0.5 * (1 - l - r)
        return R.TLegend(c - 0.5 * w, 1 - t - o - h, c + 0.5 * w, 1 - t - o, "", "NBNDC")
    if pos == 3:
        return R.TLegend(1 - r - ho - w, 1 - t - o - h, 1 - r - ho, 1 - t - o, "", "NBNDC")
    if pos == 4:
        return R.TLegend(l + ho, b + o, l + ho + w, b + o + h, "", "NBNDC")
    if pos == 5:
        c = l + 0.5 * (1 - l - r)
        return R.TLegend(c - 0.5 * w, b + o, c + 0.5 * w, b + o + h, "", "NBNDC")
    if pos == 6:
        return R.TLegend(1 - r - ho - w, b + o, 1 - r - ho, b + o + h, "", "NBNDC")


def DrawTitle(pad, text, align, textOffset=0.2, textSize=0.6):
    pad_backup = R.gPad
    pad.cd()
    t = pad.GetTopMargin()
    l = pad.GetLeftMargin()
    r = pad.GetRightMargin()

    pad_ratio = (float(pad.GetWh()) * pad.GetAbsHNDC()) / (float(pad.GetWw()) * pad.GetAbsWNDC())
    if pad_ratio < 1.0:
        pad_ratio = 1.0

    latex = R.TLatex()
    latex.SetNDC()
    latex.SetTextAngle(0)
    latex.SetTextColor(R.kBlack)
    latex.SetTextFont(42)
    latex.SetTextSize(textSize * t * pad_ratio)

    y_off = 1 - t + textOffset * t
    if align == 1:
        latex.SetTextAlign(11)
    if align == 1:
        latex.DrawLatex(l, y_off, text)
    if align == 2:
        latex.SetTextAlign(21)
    if align == 2:
        latex.DrawLatex(l + (1 - l - r) * 0.5, y_off, text)
    if align == 3:
        latex.SetTextAlign(31)
    if align == 3:
        latex.DrawLatex(1 - r, y_off, text)
    pad_backup.cd()


def StyleLimitBand(graph_dict, overwrite_style_dict=None):
    style_dict = {"obs": {"LineWidth": 2}, "exp0": {"LineWidth": 2, "LineColor": R.kRed}, "exp1": {"FillColor": R.kGreen}, "exp2": {"FillColor": R.kYellow}}
    if overwrite_style_dict is not None:
        for key in overwrite_style_dict:
            if key in style_dict:
                style_dict[key].update(overwrite_style_dict[key])
            else:
                style_dict[key] = overwrite_style_dict[key]
    for key in graph_dict:
        Set(graph_dict[key], **style_dict[key])


def DrawLimitBand(pad, graph_dict, draw=["exp2", "exp1", "exp0", "obs"], draw_legend=None, legend=None, legend_overwrite=None):
    legend_dict = {
        "obs": {"Label": "Observed", "LegendStyle": "LP", "DrawStyle": "PLSAME"},
        "exp0": {"Label": "Expected", "LegendStyle": "L", "DrawStyle": "LSAME"},
        "exp1": {"Label": "#pm1#sigma Expected", "LegendStyle": "F", "DrawStyle": "3SAME"},
        "exp2": {"Label": "#pm2#sigma Expected", "LegendStyle": "F", "DrawStyle": "3SAME"},
    }
    if legend_overwrite is not None:
        for key in legend_overwrite:
            if key in legend_dict:
                legend_dict[key].update(legend_overwrite[key])
            else:
                legend_dict[key] = legend_overwrite[key]
    pad.cd()
    for key in draw:
        if key in graph_dict:
            graph_dict[key].Draw(legend_dict[key]["DrawStyle"])
    if legend is not None:
        if draw_legend is None:
            draw_legend = reversed(draw)
        for key in draw_legend:
            if key in graph_dict:
                legend.AddEntry(graph_dict[key], legend_dict[key]["Label"], legend_dict[key]["LegendStyle"])
