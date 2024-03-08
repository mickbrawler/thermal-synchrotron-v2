'''Module containing utility functions

This file can be imported as a module and contains the following functions:
    * text_on_line - add text on a matplotlib line
    * round_to_1 - round to nearest integer
'''

import numpy as np

def text_on_line(line,ax, txt='text', fontsize=12, loc=0.5, loc_fmt='scaled', va='center', color='nan', arrow={}):
    '''Adds text to along given line.

    Parameters
    ----------
    line : matplotlib line object
            line to annotate
    ax : matplotlib axes object
            axis in which to draw
    txt : string, optional
            annotation text. default is 'text'
    loc : number, optional
            location of string along line [0-1]. default is 0.5
    loc_fmt : 'scaled','x', optional
            format for string location (normalized or physical coordinates). default is 'scaled'
    fontsize : number, optional
            text font size, default is 12
    va : 'center','top','bottom', optional
            vertical alignment of text wrt line. default is 'center'
    out : matplotlib text object'''

    x,y = line.get_xdata(),line.get_ydata()
    if color=='nan':
        color = line.get_color()

    xlim=np.array(ax.get_xlim())
    ylim=np.array(ax.get_ylim())

    # only account for portion within axes
    flt = (x>xlim[0])*(x<xlim[1])*(y>ylim[0])*(y<ylim[1])
    if sum(flt)>1:
        x = x[flt]
        y = y[flt]

    ## scale data by axis lim
    if ax.get_xscale()=='linear':
        xscaled=x/(xlim[1]-xlim[0])
        txt_x = min(x) + loc*(max(x)-min(x))
        if loc_fmt=='x':
            txt_x = loc
    elif ax.get_xscale()=='log':
        flt = (x>0) & (y>0); x,y = x[flt],y[flt] # hack to remove non-positive data (not best solution)
        xscaled=np.log10(x)/(np.log10(xlim[1])-np.log10(xlim[0]))
        txt_x = 10**(np.log10(min(x)) + loc*(np.log10(max(x))-np.log10(min(x))))
        if loc_fmt=='x':
            txt_x = loc

    if ax.get_yscale()=='linear':
        yscaled=y/(ylim[1]-ylim[0])
    elif ax.get_yscale()=='log':
        yscaled=np.log10(y)/(np.log10(ylim[1])-np.log10(ylim[0]))

    if len(x)>2:
        dydx_scaled = (yscaled[2:]-yscaled[:-2])/(xscaled[2:]-xscaled[:-2])
        x_dydx = x[1:-1]
    elif len(x)==2:
        dydx_scaled = (yscaled[1:]-yscaled[:-1])/(xscaled[1:]-xscaled[:-1])
        x_dydx = 0.5*(x[1:]+x[:-1])

    txt_y = 0.0
    if np.all(np.diff(x) > 0):
        txt_y = np.interp(txt_x,x,y)
    elif np.all(np.diff(x) < 0):
        txt_y = np.interp(txt_x,np.flipud(x),np.flipud(y))
    elif np.all(np.diff(x) == 0):
        # hack for vertical lines
        if ax.get_yscale()=='linear':
            txt_y = min(y) + loc*(max(y)-min(y))
        elif ax.get_yscale()=='log':
            txt_y = 10**(np.log10(min(y)) + loc*(np.log10(max(y))-np.log10(min(y))))
            txt_x = np.interp(txt_y,y,x)
    else:
        print('error: interpolation requires monotonic x-data')

    txt_dydx_scaled = np.interp(txt_x, x_dydx,dydx_scaled)

    axis_extent=ax.get_position().extents
    figure_size=ax.get_figure().get_size_inches()
    aspectratio=(figure_size[1]/figure_size[0])*(axis_extent[3]-axis_extent[1])/(axis_extent[2]-axis_extent[0])
    rot = np.rad2deg(np.arctan(txt_dydx_scaled*aspectratio))

    txt_obj = ax.text(txt_x,txt_y, txt, fontsize=fontsize, color=color, rotation=rot, ha='center', va='center')

    if va=='center' and len(arrow)==0:
        ## background color for overlap above line
        bgcolor='none' #ax.get_facecolor()
        txt_obj.set_backgroundcolor((1,0,0,0))
        bb = txt_obj.get_window_extent(renderer=plt.gcf().canvas.get_renderer())
        verts = ax.transData.inverted().transform(bb)
        if rot<=45:
            y2 = np.ma.masked_where(((x>verts[0][0])&(x<verts[1][0])), y)
            x2 = np.ma.masked_where(((x>verts[0][0])&(x<verts[1][0])), x)
        else:
            y2 = np.ma.masked_where(((y>verts[0][1])&(y<verts[1][1])), y)
            x2 = np.ma.masked_where(((y>verts[0][1])&(y<verts[1][1])), x)
        line.set_data(x2,y2)
    else:
        bgcolor='none'

        ## shift text above/below line (offset = fontsize)
        if va=='top':
            sgn=np.array([-1.0,1.0])
        elif va=='bottom':
            sgn=np.array([1.0,-1.0])

        if len(arrow)>0:
            dist = arrow['dist']
        else:
            dist = 1

        point_xy = ax.transData.transform((txt_x,txt_y))
        point_xy = point_xy + dist*fontsize*sgn*np.array([np.sin(np.deg2rad(rot)), np.cos(np.deg2rad(rot))])
        data_xy = ax.transData.inverted().transform(point_xy)
        txt_obj.set_position(data_xy)

        if len(arrow)>0:
            arrow.pop('dist')
            arrow['arrowstyle'] = '<-'
            arrow['color'] = color
            ax.annotate('',(txt_x,txt_y),xytext=(data_xy[0],data_xy[1]),arrowprops=arrow)

    ## set text color (= linecolor)
    txt_obj.set_backgroundcolor(bgcolor)
    txt_obj.set_clip_on('True')

    return txt_obj

def round_to_1(x):
    val = round(x, -int(np.floor(np.log10(abs(x)))))
    if np.abs(val) >= 1.0:
        val = int(val)
    return val
