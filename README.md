# A generalised form of the Transformed Eulerian Mean

[![DOI](https://zenodo.org/badge/548557109.svg)](https://zenodo.org/badge/latestdoi/548557109)

This repository contains the python code to compute generalized TEM diagnostics defined by Greatbatch et al. (2022). In addition the software allows a diagnosis of the standard TEM defined by Andrews et al. (1987) and the modified TEM defined by Held and Schneider (1999).

## TEM diagnosis from meteorological data

The generalized TEM is defined for spherical, log-pressure coordinates. Meteorological data (e.g. the ERA5 global reanalsis, Hersbach et al. 2020) is usually given on pressure levels \[hPa\] with latitude-longitude coordinates \[° N\]. The software takes full account for this. NetCDF files can be read using the [xarray](https://docs.xarray.dev/en/stable/) N-D labled arrays and datasets. All diagnostics in SI-units are computed from zonal and meridonal velocity \[m/s\], vertical velocity \[Pa/s\] and in-situ temperature [K] without changing the coordinate variables. A simple python code using this software might look like this.


```python
import xarray as xr 
from generalized_TEM import Generalized

ds = xr.open_dataset('era5-pressure-levels.nc')

tem = Generalized(ds['t'],('longitude','time'))

diffusivity = tem.diffusivity(ds['v'],ds['w'],ds['t'])

diffusivity.plot()
```


## References

- Andrews, D. G., Holton, J. R. and Leovy, C. B. (1987) Middle Atmosphere Dynamics. No. 40. Academic Press.

- Held, I. M. and Schneider, T. (1999) The surface branch of the zonally averaged mass transport circulation in the troposphere. Journal of the Atmospheric Sciences, 56, 1688–1697.

- Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D. et al. (2020) The era5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146, 1999–2049.
