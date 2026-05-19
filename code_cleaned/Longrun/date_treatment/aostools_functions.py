#!/usr/bin/python
#
# Code by Martin Jucker, distributed under an GPLv3 License
#
# This file contains parts of aostools.climate found at https://github.com/mjucker/aostools

############################################################################################
#
from __future__ import print_function
import numpy as np

## helper function: Get actual width and height of axes
def GetAxSize(fig,ax,dpi=False):
	"""get width and height of a given axis.
	   output is in inches if dpi=False, in dpi if dpi=True
	"""
	bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
	width, height = bbox.width, bbox.height
	if dpi:
		width *= fig.dpi
		height *= fig.dpi
	return width, height


# helper function: re-arrange array dimensions
def AxRoll(x,ax,invert=False):
	"""Re-arrange array x so that axis 'ax' is first dimension.
		Undo this if invert=True
	"""
	if ax < 0:
		n = len(x.shape) + ax
	else:
		n = ax
	#
	if invert is False:
		y = np.rollaxis(x,n,0)
	else:
		y = np.rollaxis(x,0,n+1)
	return y

##helper functions
def GetAnomaly(x,axis=-1):
	"""Computes the anomaly of array x along dimension axis.

	INPUTS:
	  x    - array to compute anomalies from
	  axis - axis along dimension for anomalies
	OUTPUTS:
	  x    - anomalous array
	"""    #bring axis to the front
	xt= AxRoll(x,axis)
	#compute anomalies
	xt = xt - xt.mean(axis=0)[np.newaxis,:]
	#bring axis back to where it was
	x = AxRoll(xt,axis,invert=True)
	return x

#
def ComputeVertEddy(v,t,p,p0=1e3,wave=-1):
	""" Computes the vertical eddy components of the residual circulation,
		bar(v'Theta'/Theta_p). Either in real space, or a given wave number.
		Dimensions must be time x pres x lat x lon.
		Output dimensions are: time x pres x lat
		Output units are [v_bar] = [v], [t_bar] = [v*p]

		INPUTS:
			v    - meridional wind
			t    - temperature
			p    - pressure coordinate
			p0   - reference pressure for potential temperature
			wave - wave number (if >=0)
		OUPUTS:
			v_bar - zonal mean meridional wind [v]
			t_bar - zonal mean vertical eddy component <v'Theta'/Theta_p> [v*p]
	"""
	#
	# some constants
	kappa = 2./7
	#
	# pressure quantitites
	pp0 = (p0/p[np.newaxis,:,np.newaxis,np.newaxis])**kappa
	dp  = np.gradient(p)[np.newaxis,:,np.newaxis]
	# convert to potential temperature
	t = t*pp0 # t = theta
	# zonal means
	v_bar = np.nanmean(v,axis=-1)
	t_bar = np.nanmean(t,axis=-1) # t_bar = theta_bar
	# prepare pressure derivative
	dthdp = np.gradient(t_bar,edge_order=2)[1]/dp # dthdp = d(theta_bar)/dp
	dthdp[dthdp==0] = np.nan
	# time mean of d(theta_bar)/dp
	dthdp = np.nanmean(dthdp,axis=0)[np.newaxis,:]
	# now get wave component
	if isinstance(wave,list):
		t = np.sum(GetWaves(v,t,wave=-1,do_anomaly=True)[:,:,:,wave],axis=-1)
	elif wave < 0:
		v = GetAnomaly(v) # v = v'
		t = GetAnomaly(t) # t = t'
		t = np.nanmean(v*t,axis=-1) # t = bar(v'Th')
	else:
		t = GetWaves(v,t,wave=wave,do_anomaly=True) # t = bar(v'Th'_{k=wave})
	t_bar = t/dthdp # t_bar = bar(v'Th')/(dTh_bar/dp)
	#
	return v_bar,t_bar


##############################################################################################
def ComputeEPfluxDiv(lat,pres,u,v,t,w=None,do_ubar=False,wave=-1):
	""" Compute the EP-flux vectors and divergence terms.

		The vectors are normalized to be plotted in cartesian (linear)
		coordinates, i.e. do not include the geometric factor a*cos\phi.
		Thus, ep1 is in [m2/s2], and ep2 in [hPa*m/s2].
		The divergence is in units of m/s/day, and therefore represents
		the deceleration of the zonal wind. This is actually the quantity
		1/(acos\phi)*div(F).

	INPUTS:
	  lat  - latitude [degrees]
	  pres - pressure [hPa]
	  u    - zonal wind, shape(time,p,lat,lon) [m/s]
	  v    - meridional wind, shape(time,p,lat,lon) [m/s]
	  t    - temperature, shape(time,p,lat,lon) [K]
	  w    - pressure velocity, optional, shape(time,p,lat,lon) [hPa/s]
	  do_ubar - compute shear and vorticity correction? optional
	  wave - only include this wave number. all if <0, sum over waves if a list. optional
	OUTPUTS:
	  ep1  - meridional EP-flux component, scaled to plot in cartesian [m2/s2]
	  ep2  - vertical   EP-flux component, scaled to plot in cartesian [hPa*m/s2]
	  div1 - horizontal EP-flux divergence, divided by acos\phi [m/s/d]
	  div2 - horizontal EP-flux divergence , divided by acos\phi [m/s/d]
	"""
	# some constants
	Rd    = 287.04
	cp    = 1004
	kappa = Rd/cp
	p0    = 1000
	Omega = 2*np.pi/(24*3600.) # [1/s]
	a0    = 6.371e6
	# geometry
	pilat = lat*np.pi/180
	dphi  = np.gradient(pilat)[np.newaxis,np.newaxis,:]
	coslat= np.cos(pilat)[np.newaxis,np.newaxis,:]
	sinlat= np.sin(pilat)[np.newaxis,np.newaxis,:]
	R     = 1./(a0*coslat)
	f     = 2*Omega*sinlat
	pp0  = (p0/pres[np.newaxis,:,np.newaxis])**kappa
	dp    = np.gradient(pres)[np.newaxis,:,np.newaxis]
	#
	# absolute vorticity
	if do_ubar:
		ubar = np.nanmean(u,axis=-1)
		fhat = R*np.gradient(ubar*coslat,edge_order=2)[-1]/dphi
	else:
		fhat = 0.
	fhat = f - fhat # [1/s]
	#
	## compute thickness weighted heat flux [m.hPa/s]
	vbar,vertEddy = ComputeVertEddy(v,t,pres,p0,wave) # vertEddy = bar(v'Th'/(dTh_bar/dp))
	#
	## get zonal anomalies
	u = GetAnomaly(u)
	v = GetAnomaly(v)
	if isinstance(wave,list):
		upvp = np.sum(GetWaves(u,v,wave=-1)[:,:,:,wave],-1)
	elif wave<0:
		upvp = np.nanmean(u*v,axis=-1)
	else:
		upvp = GetWaves(u,v,wave=wave)
	#
	## compute the horizontal component
	if do_ubar:
		shear = np.gradient(ubar,edge_order=2)[1]/dp # [m/s.hPa]
	else:
		shear = 0.
	ep1_cart = -upvp + shear*vertEddy # [m2/s2 + m/s.hPa*m.hPa/s] = [m2/s2]
	#
	## compute vertical component of EP flux.
	## at first, keep it in Cartesian coordinates, ie ep2_cart = f [v'theta'] / [theta]_p + ...
	#
	ep2_cart = fhat*vertEddy # [1/s*m.hPa/s] = [m.hPa/s2]
	if w is not None:
		w = GetAnomaly(w) # w = w' [hPa/s]
		if isinstance(wave,list):
			w = sum(GetWaves(u,w,wave=wave)[:,:,:,wave],-1)
		elif wave<0:
			w = np.nanmean(w*u,axis=-1) # w = bar(u'w') [m.hPa/s2]
		else:
			w = GetWaves(u,w,wave=wave) # w = bar(u'w') [m.hPa/s2]
		ep2_cart = ep2_cart - w # [m.hPa/s2]
	#
	#
	# We now have to make sure we get the geometric terms right
	# With our definition,
	#  div1 = 1/(a.cosphi)*d/dphi[a*cosphi*ep1_cart*cosphi],
	#    where a*cosphi comes from using cartesian, and cosphi from the derivative
	# With some algebra, we get
	#  div1 = cosphi d/d phi[ep1_cart] - 2 sinphi*ep1_cart
	div1 = coslat*np.gradient(ep1_cart,edge_order=2)[-1]/dphi - 2*sinlat*ep1_cart
	# Now, we want acceleration, which is div(F)/a.cosphi [m/s2]
	div1 = R*div1 # [m/s2]
	#
	# Similarly, we want acceleration = 1/a.coshpi*a.cosphi*d/dp[ep2_cart] [m/s2]
	div2 = np.gradient(ep2_cart,edge_order=2)[1]/dp # [m/s2]
	#
	# convert to m/s/day
	div1 = div1*86400
	div2 = div2*86400
	#
	return ep1_cart,ep2_cart,div1,div2

##############################################################################################
def PlotEPfluxArrows(x,y,ep1,ep2,fig,ax,xlim=None,ylim=None,xscale='linear',yscale='linear',invert_y=True, newax=False, pivot='tail',scale=None):
	"""Correctly scales the Eliassen-Palm flux vectors for plotting on a latitude-pressure or latitude-height axis.
		x,y,ep1,ep2 assumed to be xarray.DataArrays.

	INPUTS:
		x       : horizontal coordinate, assumed in degrees (latitude) [degrees]
		y       : vertical coordinate, any units, but usually this is pressure or height
		ep1     : horizontal Eliassen-Palm flux component, in [m2/s2]. Typically, this is ep1_cart from
		           ComputeEPfluxDiv()
		ep2     : vertical Eliassen-Palm flux component, in [U.m/s2], where U is the unit of y.
			       Typically, this is ep2_cart from ComputeEPfluxDiv(), in [hPa.m/s2] and y is pressure [hPa].
		fig     : a matplotlib figure object. This figure contains the axes ax.
		ax      : a matplotlib axes object. This is where the arrows will be plotted onto.
		xlim    : axes limits in x-direction. If None, use [min(x),max(x)]. [None]
		ylim    : axes limits in y-direction. If None, use [min(y),max(y)]. [None]
		xscale  : x-axis scaling. currently only 'linear' is supported. ['linear']
		yscale  : y-axis scaling. 'linear' or 'log' ['linear']
		invert_y: invert y-axis (for pressure coordinates). [True]
		newax   : plot on second y-axis. [False]
		pivot   : keyword argument for quiver() ['tail']
		scale   : keyword argument for quiver() [None]

	OUTPUTS:
	   Fphi*dx : x-component of properly scaled arrows. Units of [m3.inches]
	   Fp*dy   : y-component of properly scaled arrows. Units of [m3.inches]
	   ax   : secondary y-axis if newax == True
	"""
	import numpy as np
	import matplotlib.pyplot as plt
	#
	def Deltas(z,zlim):
		if zlim is None:
			return np.max(z)-np.min(z)
		else:
			return zlim[1]-zlim[0]
	# Scale EP vector components as in Edmon, Hoskins & McIntyre JAS 1980:
	cosphi = np.cos(np.deg2rad(x))
	a0 = 6376000.0 # Earth radius [m]
	grav = 9.81
	# first scaling: Edmon et al (1980), Eqs. 3.1 & 3.13
	Fphi = 2*np.pi/grav*cosphi**2*a0**2*ep1 # [m3.rad]
	Fp   = 2*np.pi/grav*cosphi**2*a0**3*ep2 # [m3.hPa]
	#
	# Now comes what Edmon et al call "distances occupied by 1 radian of
	#  latitude and 1 [hecto]pascal of pressure on the diagram."
	# These distances depend on figure aspect ratio and axis scale
	#
	# first, get the axis width and height for
	#  correct aspect ratio
	width,height = GetAxSize(fig,ax)
	# we use min(),max(), but note that if the actual axis limits
	#  are different, this will be slightly wrong.
	delta_x = Deltas(x,xlim)
	delta_y = Deltas(y,ylim)
	#
	#scale the x-axis:
	if xscale == 'linear':
		dx = width/delta_x/np.pi*180
	else:
		raise ValueError('ONLY LINEAR X-AXIS IS SUPPORTED AT THE MOMENT')
	#scale the y-axis:
	if invert_y:
		y_sign = -1
	else:
		y_sign = 1
	if yscale == 'linear':
		dy = y_sign*height/delta_y
	elif yscale == 'log':
		dy = y_sign*height/y/np.log(np.max(y)/np.min(y))
	#
	# plot the arrows onto axis
	quivArgs = {'angles':'uv','scale_units':'inches','pivot':pivot}
	if scale is not None:
		quivArgs['scale'] = scale
	if newax:
		ax = ax.twinx()
		ax.set_ylabel('pressure [hPa]')
	try:
		ax.quiver(x,y,Fphi*dx,Fp*dy,**quivArgs)
	except:
		ax.quiver(x,y,Fphi.transpose()*dx,Fp.transpose()*dy,**quivArgs)
	if invert_y:
		ax.invert_yaxis()
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)
	ax.set_yscale(yscale)
	ax.set_xscale(xscale)
	#
	if newax:
		return Fphi*dx,Fp*dy,ax
	else:
		return Fphi*dx,Fp*dy

##############################################################################################
def GetWaves(x,y=None,wave=-1,axis=-1,do_anomaly=False):
	"""Get Fourier mode decomposition of x, or <x*y>, where <.> is zonal mean.

		If y!=[], returns Fourier mode contributions (amplitudes) to co-spectrum zonal mean of x*y. Shape is same as input, except axis which is len(axis)/2+1 due to Fourier symmetry for real signals.

		If y=[] and wave>=0, returns real space contribution of given wave mode. Output has same shape as input.
		If y=[] and wave=-1, returns real space contributions for all waves. Output has additional first dimension corresponding to each wave.

	INPUTS:
		x          - the array to decompose
		y          - second array if wanted
		wave       - which mode to extract. all if <0
		axis       - along which axis of x (and y) to decompose
		do_anomaly - decompose from anomalies or full data
	OUTPUTS:
		xym        - data in Fourier space
	"""
	initShape = x.shape
	x = AxRoll(x,axis)
	if y is not None:
		y = AxRoll(y,axis)
	# compute anomalies
	if do_anomaly:
		x = GetAnomaly(x,0)
		if y is not None:
			y = GetAnomaly(y,0)
	# Fourier decompose
	x = np.fft.fft(x,axis=0)
	nmodes = x.shape[0]/2+1
	if wave < 0:
			if y is not None:
				xym = np.zeros((nmodes,)+x.shape[1:])
			else:
				xym = np.zeros((nmodes,)+initShape)
	else:
		xym = np.zeros(initShape[:-1])
	if y is not None:
			y = np.fft.fft(y,axis=0)
			# Take out the waves
			nl  = x.shape[0]**2
			xyf  = np.real(x*y.conj())/nl
			# due to symmetric spectrum, there's a factor of 2, but not for wave-0
			mask = np.zeros_like(xyf)
			if wave < 0:
				for m in range(xym.shape[0]):
					mask[m,:] = 1
					mask[-m,:]= 1
					xym[m,:] = np.sum(xyf*mask,axis=0)
					mask[:] = 0
				xym = AxRoll(xym,axis,invert=True)
			else:
				xym = xyf[wave,:]
				if wave >= 0:
					xym = xym + xyf[-wave,:]
	else:
			mask = np.zeros_like(x)
			if wave >= 0:
				mask[wave,:] = 1
				mask[-wave,:]= 1 # symmetric spectrum for real signals
				xym = np.real(np.fft.ifft(x*mask,axis=0))
				xym = AxRoll(xym,axis,invert=True)
			else:
				for m in range(xym.shape[0]):
					mask[m,:] = 1
					mask[-m,:]= 1 # symmetric spectrum for real signals
					fourTmp = np.real(np.fft.ifft(x*mask,axis=0))
					xym[m,:] = AxRoll(fourTmp,axis,invert=True)
					mask[:] = 0
	return np.squeeze(xym)

##helper functions
def GetAnomaly(x,axis=-1):
	"""Computes the anomaly of array x along dimension axis.

	INPUTS:
	  x    - array to compute anomalies from
	  axis - axis along dimension for anomalies
	OUTPUTS:
	  x    - anomalous array
	"""    #bring axis to the front
	xt= AxRoll(x,axis)
	#compute anomalies
	xt = xt - xt.mean(axis=0)[np.newaxis,:]
	#bring axis back to where it was
	x = AxRoll(xt,axis,invert=True)
	return x
