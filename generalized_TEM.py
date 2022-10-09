'''
    Compute Transformed Eulerian Mean Diagnostics for
    
    - the generalized TEM defined by Greatbatch et al. (2022)
    - the standard TEM defined by Andrews et al. (1987)
    - the modified TEM defined by Held & Schneider (1999)
'''
import xarray as xr
import numpy as np
import warnings
import math


class TEM:
    '''
        Parent class for a Transformed Eulerian Mean
        
        Attributes
        ----------
        lev_name : str
            name of pressure coordinate
        phi_name : str
            name of meridional coordinate
        phi : numpy.array
            meridional coordinate in radians
        cos_phi : numpy.array
            cosinus of meridional coordiante
        coriolis : numpy.array
            Coriolis parameter
        a : float
           earth radius
        H : float
            scale height
        lev : numpy.array
            log-pressure coordinate
        rho_0 : numpy.array
            mean density
        to_log_pressure : lambda
            convert omega to units of  m/s
        potential_temperature : lambda
            convert in-situ temperature to potential temperature
        dims : tuple
            dimensions to reduce
        Tm : numpy.array
            mean in-situ temperature
        theta : numpy.array
            mean potential temperature
            
        Methods
        -------
        grad(self,da)
            Compute gradient of scalar field
        rot_grad(self,da)
            Compute rotated gradient of scalar field
        spher_div(self,A,B)
            Compute divergence of vector field
        tracer_moment_flux(self,u,T,order=1)
            Compute flux as product of full velocity and tracer anomaly
        Jn(self,v,omega,T,order=1)
            Compute isopycnal tracer moment flux
        streamfunction(self,v)
            Compute streamfunction of non-divergent vector field by vertical integration
        momentum_flux(self,u,v,omega)
            Compute eddy angular momentum flux
        angular(self,u)
            Compute negative gradient of angular momentum
        EP_flux(self,u,v,omega,T,order=None)
            Compute the Eliassen-Palm flux
    '''
    
    def __init__(self,T,
                 dims=('longitude',),
                 H=None,ps=1013,a=6371000):
        '''
            Parameters
            ----------
            T : xarray.DataArray
                in-situ temperature [K]
            dims : tuple, optional
                dimensions to reduce
            H : float, optional
                scale height [m] (default is computed from in-situ temperature)
            ps : int, optional
                surface pressure [hPa] (default is 1013)
            a : int, optional
                earth radius (default is 6371000)
        '''
        coords = T.dims  
        
        # pressure coordinate 
        if 'level' in coords:
            self.lev_name = 'level'
        elif 'lev' in coords:
            self.lev_name = 'lev'
        elif 'plev' in coords:
            self.lev_name = 'plev'
        elif 'pressure' in coords:
            self.lev_name = 'pressure'
        else:
            raise KeyError('Unknown pressure coordinate')
        lev = T[self.lev_name]
            
            
        # meridional coordinate
        if 'latitude' in coords:
            self.phi_name = 'latitude'
        elif 'lat' in coords:
            self.phi_name = 'lat'
        else:
            raise KeyError('Unknown meridional coordinate')
        phi = T[self.phi_name]
            
            
        # check coordinate units
        if not(T[self.lev_name].attrs['units'] in ['millibars','hPa']):
            warnings.warn('Unable to validate units of pressure coordinate')
        if not(T[self.phi_name].attrs['units'] in ['degrees_north',]):
            warnings.warn('Unable to validate units of meridional coordinate')
            
            
        # transform coordinates to log pressure and radians
        self.phi = np.radians(phi)
        self.cos_phi = np.cos(self.phi)
        self.coriolis = 4 * np.pi / 86400 * np.sin(self.phi)
        self.a = a
        
        # compute scale height from mean temperature if not specified by kwargs
        R = 8.314
        g = 9.81
        M = 0.029
        if H is None:
            self.H = T.mean().values * R / (M * g)
        else:
            self.H = H
        self.lev = - self.H * np.log(lev / ps)
        
        # mean density accounted for compressibility
        rho_s = ps / (self.H * g) * 100 
        self.rho_0 = rho_s * np.exp(-self.lev / self.H)
        
        # transform omega in Pa/s to w in m/s
        self.to_log_p = lambda omega: omega * (- self.H / lev) / 100
        
        # compute potential temperature
        kappa = 0.2854
        self.potential_temperature = lambda T: T * np.exp(np.log(ps / lev) * kappa)
        
        # temperature mean
        self.dims = dims
        self.Tm = T.mean(dims).compute()
        self.theta = self.potential_temperature(self.Tm)
        
 
    def grad(self,da):
        '''
            Compute gradient of scalar field
            
            Parameters
            ----------
            da : xarray.DataArray
                scalar fiedl
                
            Returns
            -------
            xarray.DataArray
                meridional vector component
            xarray.DataArray
                vertical vector compomemt
        '''
        A = 1 / self.a * da.differentiate(self.phi_name) / self.phi.differentiate(self.phi_name)
        B = da.differentiate(self.lev_name) / self.lev.differentiate(self.lev_name)
        return A, B
        
        
    def rot_grad(self,da):
        '''
            Compute rotated gradient of scalar field
                        
            Parameters
            ----------
            da : xarray.DataArray
                scalar field
                
            Returns
            -------
            xarray.DataArray
                meridional vector component
            xarray.DataArray
                vertical vector compomemt
        '''
        A = - 1 / self.rho_0 / self.cos_phi * da.differentiate(self.lev_name) / self.lev.differentiate(self.lev_name)
        B = 1 / self.a / self.rho_0 / self.cos_phi * da.differentiate(self.phi_name) / self.phi.differentiate(self.phi_name)
        return A, B
        
        
    def spher_div(self,A,B):
        '''
            Compute divergence of vector field
            
            Parameters
            ----------
            A : xarray.DataArray
                meridional vector component
            B : xarray.DataArray
                vertical vector component
                
            Returns
            -------
            xarray.DataArray
                divergence
        '''
        sumA = 1 / self.a / self.cos_phi * (self.cos_phi * A).differentiate(self.phi_name) / self.phi.differentiate(self.phi_name)
        sumB = B.differentiate(self.lev_name) / self.lev.differentiate(self.lev_name)
        return sumA + sumB
    
    
    def tracer_moment_flux(self,u,T,order=1):
        '''
            Compute flux as product of full velocity and tracer anomaly
            
            Parameters
            ----------
            u : xarray.DataArray
                velocity component [m/s]
            T : xarray.DataArray
                in-situ Temperature [K]
            order : int, optinal
                order of tracer moment (default is 1)
                
            Returns
            -------
            xarray.DataArray
                tracer moment flux component
        '''
        moment = (T - self.Tm) ** order
        moment = moment / order
        flux = (u * moment).mean(self.dims)
        flux = flux.persist()
        
        # turn temperature flux into potential temperature flux
        for i in range(order):
            flux = self.potential_temperature(flux)
        return flux
    
    
    def Jn(self,v,omega,T,order=1):
        '''
            Compute isopycnal tracer moment flux
            
            Parameters
            ----------
            v : xarray.DataArray
                meridional velocity [m/s]
            omega : xarray.DataArray
                vertical velocity [Pa/s]
            T : xarray.DataArray
                in-situ temperature [K]
            order : int, optinal
                order of tracer moment (default is 1)
                
            Returns
            -------
            xarray.DataArray
                isopycnal tracer moment flux component 
        '''
        A = self.rho_0 * self.cos_phi * self.tracer_moment_flux(v,T,order=order)
        w = self.to_log_p(omega)
        B = self.rho_0 * self.cos_phi * self.tracer_moment_flux(w,T,order=order)
        projection = self.isopycnal(A,B)
        return projection
    
    
    def streamfunction(self,v):
        '''
            Compute streamfunction of non-divergent vector field by vertical integration
            
            Parameters
            ----------
            v : xarray.DataArray
                meridional velocity component [m/s]
                
            Returns
            -------
            xarray.DataArray
                mass streamfunction
            
        '''
        spacing = self.lev.values
        spacing = np.insert(np.append(spacing,-spacing[-1]),0,spacing[0])
        spacing = (spacing[:-2]-spacing[2:])/2
        
        spacing = xr.DataArray(spacing,
                               dims=[self.lev_name],
                               coords={self.lev_name:self.lev[self.lev_name]})
        
        transport = v.mean(self.dims).compute()
        transport = - transport * spacing * self.rho_0 * self.cos_phi
        transport = transport.cumsum(self.lev_name)
        
        return transport
    
    
    def momentum_flux(self,u,v,omega):
        '''
            Compute eddy angular momentum flux
            
            Parameters
            ----------
            u : xarray.DataArray
                zonal velocity component [m/s]
            v : xarray.DataArray
                meridional velocity component [m/s]
            omega : xarray.DataArray
                vertical velocity component [Pa/s]
                
            Returns
            -------
            xarray.DataArray
                meridional flux component
            xarray.DataArray
                vertical flux component
        '''
        anomal = u - u.mean(self.dims)
        
        yflux = (anomal * v).mean(self.dims)
        yflux = self.rho_0 * self.cos_phi * yflux
        
        zflux = (anomal * omega).mean(self.dims)
        zflux = self.to_log_p(zflux)
        zflux = self.rho_0 * self.cos_phi * zflux
        
        return yflux, zflux
    
    
    def angular(self,u):
        '''
            Compute negative gradient of angular momentum
            
            Parameters
            ----------
            u : xarray.DataArray
                zonal velocity component [m/s]
                
            Returns
            -------
            xarray.DataArray
                meridonal component
            xarray.DataArray
                vertical component
        '''
        momentum = -self.cos_phi * u.mean(self.dims)
        A, B = self.grad(momentum)
        A = A / self.cos_phi
        B = B / self.cos_phi
        A = A + self.coriolis
        
        return A, B
    
    
    def EP_flux(self,u,v,omega,T,order=None):
        '''
            Compute the Eliassen-Palm flux
            
            Parameters
            ----------
            u : xarray.DataArray
                zonal velocity component [m/s]
            v : xarray.DataArray
                meridional velocity component [m/s]
            omega : xarray.DataArray
                vertical velocity component [Pa/s]
            T : xarray.DataArray
                in-situ temperature [K]
            order : int, optional
                order of generalized TEM (default is None)
                
            Returns
            -------
            xarray.DataArray
                meridonal component
            xarray.DataArray
                vertical component
        '''
        yflux, zflux = self.momentum_flux(u,v,omega)
        
        A, B = self.angular(u)
        eChi = self.eChi(v,omega,T,order=order)
        yflux = -yflux + B * eChi
        zflux = -zflux - A * eChi
        
        yflux = self.a * yflux
        zflux = self.a * zflux
        
        return yflux, zflux
        
   
    
class Generalized(TEM):
    '''
        Generalize the TEM by removing rotational tracer fluxes from the definition
        of eddd-induced streamfiunction and eddy diffusivity
        
        Methods
        -------
        isopycnal(self,A,B)
            Project vector onto $\vec s$ to compute the isopycnal component
        diapycnal(self,A,B)
            Project vector onto $\vec s$ to compute the diapycnal component
        ddm(self,da)
            Compute the diapycnal normalized gradient
        rotational(self,v,omega,T,order=3)
            Diagnpose the rotational tracer flux streafunction
        eChi(self,v,omega,T,order=3)
            Diagnose the eddy induced circulation streamfunction
        diffusivity(self,v,omega,T,order=3)
            Diagnose the eddy diffusivity
    '''
    def __init__(self,T,
                 dims=('longitude',),
                 H=None,ps=1013,a=6371000):
        '''
        Parameters
            ----------
            T : xarray.DataArray
                in-situ temperature [K]
            dims : tuple, optional
                dimensions to reduce
            H : float, optional
                scale height [m] (default is computed from in-situ temperature)
            ps : int, optional
                surface pressure [hPa] (default is 1013)
            a : int, optional
                earth radius (default is 6371000)
        '''
        super().__init__(T,dims=dims,H=H,ps=ps,a=a)
    
    
    def isopycnal(self,A,B):
        '''
            Project vector onto $\vec s$ to compute the isopycnal component
            
            Parameters
            ----------
            A : xarray.DataArray
                meridional vector component
            B : xarray.DataArray
                vertical vector component
                
            Returns
            -------
            xarray.DataArray
                isopycnal component
        '''
        sA, sB = self.rot_grad(self.theta)
        projection = (sA*A + sB*B) / np.sqrt(sA**2 + sB**2)
        return projection
    
    
    def diapycnal(self,A,B):
        '''
            Project vector onto $\vec n$ to compute the diapycnal component
            
            Parameters
            ----------
            A : xarray.DataArray
                meridional vector component
            B : xarray.DataArray
                vertical vector component
                
            Returns
            -------
            xarray.DataArray
                diapycnal component
        '''
        nA, nB = self.grad(self.theta)
        projection = (nA*A + nB*B) / np.sqrt(nA**2 + nB**2)
        return projection
        
        
    def ddm(self,da):
        '''
            Compute the diapycnal normalized gradient
           
            Parameters
            ----------
            da : xarray.DataArray
                scalar field
            
            Returns
            -------
            xarray.DataArray
                diapycnal component
        '''
        nA, nB = self.grad(self.theta)
        normalized = da / np.sqrt(nA**2 + nB**2)
        A, B = self.grad(normalized)
        projection = self.diapycnal(A,B)
        return projection
    
    
    def rotational(self,v,omega,T,order=3):
        '''
            Diagnpose the rotational tracer flux streafunction
            
            Parameters
            ----------
            v : xarray.DataArray
                meridional velocity component [m/s]
            omega : xarray.DataArray
                vertical velocity component [Pa/s]
            T : xarray.DataArray
                in-situ temperature [K]
            order : int, optional
                order of generalized TEM (default is 3)
            
            Returns
            -------
            xarray.DataArray
                flux streamfunction
        '''
        # this is equation 25
        flux = 0
        for i in range(2,order+1):
            term = self.Jn(v,omega,T,order=i)
            for j in range(2,i):
                term = self.ddm(term)
            term = term / math.factorial(i-1)
            term = term * ((-1)**(i))
            flux = flux + term
        
        # divide by magnitude of mean tracer gradient to obtain tracer flux streamfunciton
        nA, nB = self.grad(self.theta)
        R = flux / np.sqrt(nA**2 + nB**2)
        
        return R
        
        
    def eChi(self,v,omega,T,order=3):
        '''
            Diagnose the eddy induced circulation streamfunction
            
            $B|\nabla\bar\theta| 
            = \rho_0\cos\phi (\overline{v'\theta'},\overline{w'\theta'})\cdot \vec s - \nabla R \cdot \vec n$
            
            Parameters
            ----------
            v : xarray.DataArray
                meridional velocity component [m/s]
            omega : xarray.DataArray
                vertical velocity component [Pa/s]
            T : xarray.DataArray
                in-situ temperature [K]
            order : int, optional
                order of generalized TEM (default is 3)
                
            Returns
            -------
            xarray.DataArray
                mass streamfunction
        '''
        # this is equation 12 or 24 
        A, B = self.grad(self.rotational(v,omega,T,order=order))
        flux = self.Jn(v,omega,T,order=1) - self.diapycnal(A,B)
        
        # divid by magnitude of mean tracer gradient
        nA, nB = self.grad(self.theta)
        B = flux / np.sqrt(nA**2 + nB**2)
        
        return B
    
    
    def diffusivity(self,v,omega,T,order=3):
        '''
            Diagnose the eddy diffusivity
            
            $K|\nabla\bar\theta| 
            = - (\overline{v'\theta'},\overline{w'\theta'})\cdot \vec n - (\rho_0 \cos\phi)^{-1}\nabla R \cdot \vec s $
            
            Parameters
            ----------
            v : xarray.DataArray
                meridional velocity component [m/s]
            omega : xarray.DataArray
                vertical velocity component [Pa/s]
            T : xarray.DataArray
                in-situ temperature [K]
            order : int, optional
                order of generalized TEM (default is 3)
                
            Returns
            -------
            xarray.DataArray
                eddy diffusivity
        '''
        # no respective equation is given in the note yet
        # but this follows from equation 10 just as equation 12 does
        A, B = self.grad(self.rotational(v,omega,T,order=order))
        flux = -self.isopycnal(A,B) / self.rho_0 / self.cos_phi
        A = self.tracer_moment_flux(v,T,order=1)
        w = self.to_log_p(omega)
        B = self.tracer_moment_flux(w,T,order=1)
        flux = flux - self.diapycnal(A,B)
        
        # divid by magnitude of mean tracer gradietn
        nA, nB = self.grad(self.theta)
        K = flux / np.sqrt(nA**2 + nB**2)
        
        return K
        


class Standard(TEM):
    '''
        Standard TEM diagnostics as defined by Andrews et al. (1987)
        assuming horizontal isopycnals.
        
        Attributes
        ----------
        isopycnal : lambda
            Compute isopycnal component of vecotor field
        
        Methods
        -------
        eChi(self,v,omega,T,order=None)
            Diagnose the eddy induced circulation streamfunction
    '''
    def __init__(self,T,
                 dims=('longitude',),
                 H=None,ps=1013,a=6371000):
        '''
            Parameters
            ----------
            T : xarray.DataArray
                in-situ temperature [K]
            dims : tuple, optional
                dimensions to reduce
            H : float, optional
                scale height [m] (default is computed from in-situ temperature)
            ps : int, optional
                surface pressure [hPa] (default is 1013)
            a : int, optional
                earth radius (default is 6371000)
        '''
        super().__init__(T,dims=dims,H=H,ps=ps,a=a)
        
        self.isopycnal = lambda A, B: A
        
        
        
    def eChi(self,v,omega,T,order=None):
        '''
            Diagnose the eddy induced circulation streamfunction
            
            Parameters
            ----------
            v : xarray.DataArray
                meridional velocity component [m/s]
            omega : xarray.DataArray
                vertical velocity component [Pa/s]
            T : xarray.DataArray
                in-situ temperature [K]
            order : int, optional
                order of generalized TEM (default is None)
        '''
        flux = self.rho_0 * self.cos_phi * self.tracer_moment_flux(v,T,order=1)
        
        # divide by the vertical component of mean tracer gradient
        nA, nB = self.grad(self.theta)
        B = - flux / nB
        
        return B



class Modified(TEM):
    '''
        Modified TEM diagnostics as defined by Held & Schneider (1999)
        
        Attributes
        ----------
        isopycnal : lambda
            Compute isopycnal component of vecotor field
        
        Methods
        -------
        eChi(self,v,omega,T,order=None)
            Diagnose the eddy induced circulation streamfunction
    '''
    def __init__(self,T,
                 dims=('longitude',),
                 H=None,ps=1013,a=6371000):
        '''
            Parameters
            ----------
            T : xarray.DataArray
                in-situ temperature [K]
            dims : tuple, optional
                dimensions to reduce
            H : float, optional
                scale height [m] (default is computed from in-situ temperature)
            ps : int, optional
                surface pressure [hPa] (default is 1013)
            a : int, optional
                earth radius (default is 6371000)
        '''
        super().__init__(T,dims=dims,H=H,ps=ps,a=a)
        
        self.isopycnal = lambda A, B: B
        
        
        
    def eChi(self,v,omega,T,order=None):
        '''
            Diagnose the eddy induced circulation streamfunction
            
            Parameters
            ----------
            v : xarray.DataArray
                meridional velocity component [m/s]
            omega : xarray.DataArray
                vertical velocity component [Pa/s]
            T : xarray.DataArray
                in-situ temperature [K]
            order : int, optional
                order of generalized TEM (default is None)
        '''
        w = self.to_log_p(omega)
        flux = self.rho_0 * self.cos_phi * self.tracer_moment_flux(w,T,order=1)
        
        # divid by horizontal component of mean tracer gradient
        nA, nB = self.grad(self.theta)
        B = flux / nA
        
        return B
        
        
        
def _get_aspect(fig,ax):
    '''
        Compute aspect ratio of axes X/Y
        
        Parameters
        ----------
        fig : object
            figure
        ax : object
            axes
        
        Returns
        -------
        float
            ratio
    '''
    pos = ax.get_position().get_points()
    size = fig.get_size_inches()
    
    aspect = (pos[1,0]-pos[0,0]) / (pos[1,1]-pos[0,1]) * size[0] / size[1]
    return aspect



def epQuiver(tem,fig,ax,u,v,scale=None,n=22):
    '''
        Produce quiver plot for EP-flux vectors 
        
        Parameters
        ----------
        tem : object
            instance of TEM class
        fig : object
            figure
        ax : object
            axes
        u : xarray.DataArray
            meridional EP-flux component
        v : xarray.DataArray
            vertical EP-flux component
        scale : float, optional
            scale for the lenght of the arrows (default is None)
        n : int, optional
            number of vectors in each dimension (default is 22)
    '''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # scale vectors following Jucker (2021)
    uhat = 2*np.pi/tem.rho_0 * tem.a*tem.cos_phi * u
    vhat = 2*np.pi/tem.rho_0 * tem.a*tem.a*tem.cos_phi * v
    
    # for latitude in radians
    #ux = uhat /(xlim[1]-xlim[0])
    # for latitude in degrees
    ux = uhat /(xlim[1]-xlim[0]) / np.pi * 180
    ux = ux * _get_aspect(fig,ax)
    # for level in log_pressure
    #vy = vhat / (ylim[1]-ylim[0])
    # for level in pressure
    vy = vhat / (-tem.H * np.log(ylim[1]/ylim[0]))
    
    # interpolate to regular grid or logarithmic grid
    X = np.linspace(xlim[0],xlim[1],n)
    X = X[1:-1]
    #Y = np.linspace(ylim[0],ylim[1],n)
    Y = np.exp(np.linspace(np.log(ylim[0]),np.log(ylim[1]),n))
    Y = Y[1:-1]
    
    U = ux.interp(latitude=X,level=Y)
    V = vy.interp(latitude=X,level=Y)
    
    q = ax.quiver(X,Y,U,V,scale=scale,color='black')