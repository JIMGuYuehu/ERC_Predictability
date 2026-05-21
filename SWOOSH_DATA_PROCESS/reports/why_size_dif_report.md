# SWOOSH NetCDF Size Difference Diagnostics

## File Size Summary

| file | data model | disk format | size MiB | raw payload MiB | file/raw |
|---|---:|---:|---:|---:|---:|
| reference | NETCDF4 | HDF5 | 40.87 | 48.73 | 0.839 |
| original_processed | NETCDF4 | HDF5 | 16.14 | 48.73 | 0.331 |
| filled | NETCDF4 | HDF5 | 16.15 | 48.73 | 0.331 |
| filled_netcdf4_classic | NETCDF4_CLASSIC | HDF5 | 16.15 | 48.73 | 0.331 |

## `vmro3` Storage

| file | dtype | raw MiB | compressed chunk MiB | mean chunk ratio | filters / chunking |
|---|---:|---:|---:|---:|---|
| reference | float32 | 48.73 | 40.83 | 0.838 | zlib=True, complevel=9, shuffle=False, fletcher32=False, chunking=[7, 33, 48, 72] |
| original_processed | float32 | 48.73 | 16.10 | 0.330 | zlib=True, complevel=9, shuffle=False, fletcher32=False, chunking=[7, 33, 48, 72] |
| filled | float32 | 48.73 | 16.11 | 0.331 | zlib=True, complevel=9, shuffle=False, fletcher32=False, chunking=[7, 33, 48, 72] |
| filled_netcdf4_classic | float32 | 48.73 | 16.11 | 0.331 | zlib=True, complevel=9, shuffle=False, fletcher32=False, chunking=[7, 33, 48, 72] |

## Coordinate/Bounds Equality vs Reference

| file | variable order matches? | filters/chunking all match? | coordinates/bounds exact? | max abs diff summary |
|---|---:|---:|---:|---|
| original_processed | False | True | True | {} |
| filled | False | True | True | {} |
| filled_netcdf4_classic | True | True | True | {} |

## Conclusion

The CMIP6 reference and SWOOSH files both store vmro3 as float32, so the 41 MB vs 17 MB difference is not caused by double vs float32 precision in vmro3. The common variables also use the same zlib level, shuffle setting, fletcher32 setting, and chunking as the reference. The remaining size difference is therefore due to HDF5/NetCDF4 compressibility of the actual data values: the SWOOSH ozone field compresses substantially better than the CMIP6 reference field.

The reference CMIP6 example and the SWOOSH files all use `float32` for `vmro3`. The new NETCDF4_CLASSIC file was written with the reference file's chunking and compression filters for the common variables.

## Classic Output

`/mnt/soclim0/public_data/weiji/swoosh/SWOOSH_nan_fill_20260521/vmro3_SWOOSHv02.72_combinedo3q_neg2miss_remapped_to_CMIP6_template_201912-202101_filled_no_missing_netcdf4_classic.nc`
