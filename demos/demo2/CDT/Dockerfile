# Climate Data Tools (CDT) GUI container.
# r2u provides CRAN packages as Ubuntu binaries (amd64 + arm64).
FROM rocker/r2u:noble

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    HOME=/home/cdt

# System libraries: Tcl/Tk + Tktable/BWidget for the GUI, GDAL/NetCDF for data,
# X11 client bits, and a VNC/noVNC stack for the optional browser mode.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tcl-dev tk-dev tk-table bwidget \
        libgdal-dev gdal-bin libgeos-dev libproj-dev libudunits2-dev \
        libnetcdf-dev netcdf-bin \
        libx11-6 libxext6 libxrender1 libxft2 xauth x11-utils fonts-dejavu \
        tigervnc-standalone-server tigervnc-tools openbox novnc websockify \
        tini \
    && rm -rf /var/lib/apt/lists/*

# R dependencies (installed as binaries via bspm/apt).
RUN install.r \
        ncdf4 sp sf raster gstat reshape2 foreach doParallel doSNOW R.utils \
        matrixStats fields lmomco lattice latticeExtra RColorBrewer gridBase \
        jsonlite XML xml2 urltools curl httr rvest stringr stringi devtools \
        pkgbuild fitdistrplus ADGofTest qmap maps units hexbin future googledrive \
    && rm -rf /tmp/downloaded_packages \
    && sed -i 's/^suppressMessages(bspm::enable())/# &  (disabled: containers run as non-root)/' /etc/R/Rprofile.site

# CDT resolves "~" at install time, so install with the same HOME used at runtime.
COPY DESCRIPTION NAMESPACE /tmp/CDT/
COPY R /tmp/CDT/R
COPY src /tmp/CDT/src
COPY inst /tmp/CDT/inst
COPY man /tmp/CDT/man
RUN mkdir -p /home/cdt \
    && R CMD INSTALL /tmp/CDT \
    && rm -rf /tmp/CDT /home/cdt/* \
    && chmod 1777 /home/cdt

COPY docker/entrypoint.sh /usr/local/bin/cdt-entrypoint
RUN chmod +x /usr/local/bin/cdt-entrypoint

RUN mkdir -p /data && chmod 1777 /data
WORKDIR /data
EXPOSE 6080
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/cdt-entrypoint"]
CMD ["x11"]
