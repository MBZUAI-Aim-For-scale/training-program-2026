
StnChkCoordsFormatHtml <- function(){
    jsonfile <- file.path(tempdir(), "station_coords.json")
    crds <- .cdtData$EnvData$Maps.Disp
    ###
    ii <- switch(.cdtData$EnvData$output$params$data.type, "cdtstation" = 2:3, 3:4)
    nom0 <- names(crds)
    nom1 <- c(nom0[ii], 'LonX', 'LatX', 'StatusX')
    inom <- !nom0 %in% nom1

    xcrd <- crds[, c('LonX', 'LatX'), drop = FALSE]
    xcrd <- paste(xcrd[, 1], xcrd[, 2], sep = "_")
    ix1 <- duplicated(xcrd) & !is.na(crds$LonX)
    ix2 <- duplicated(xcrd, fromLast = TRUE) & !is.na(crds$LonX)
    ix <- ix1 | ix2
    icrd <- unique(xcrd[ix])
    if(length(icrd) > 0){
        for(jj in icrd){
            ic <- xcrd == jj
            xx <- apply(crds[ic, inom, drop = FALSE], 2, paste0, collapse = " | ")
            xx <- matrix(xx, nrow = 1, dimnames = list(NULL, names(xx)))
            xx <- do.call(rbind, lapply(seq_along(which(ic)), function(i) xx))
            crds[ic, inom] <- xx
        }
    }
    ###
    don <- jsonlite::toJSON(crds, pretty = TRUE)
    cat(don, file = jsonfile)

    lon.c <- mean(crds$LonX, na.rm = TRUE)
    lat.c <- mean(crds$LatX, na.rm = TRUE)
    nom <- names(crds)
    nom <- nom[!nom %in% c('LonX', 'LatX', 'StatusX', 'idx')]
    contenu <- lapply(nom, function(x) paste0("'<b>", x, " : </b>' + this.", x, " + '<br>'"))
    contenu <- do.call(paste, c(contenu, sep = " + "))
    googleAPIKey <- .cdtData$Config$Google.Maps.API.key
    if(!is.null(googleAPIKey)){
        if(length(googleAPIKey) == 0 || googleAPIKey == ""){
            googleAPIKey <- "YOUR_API_KEY"
        }
    }else{
        googleAPIKey <- "YOUR_API_KEY"
    }

    to_repl <- list(serverPath = .cdtEnv$cdtUrl,
                    lon_c = lon.c,
                    lat_c = lat.c,
                    contenu = contenu,
                    googleAPIKey = googleAPIKey)

    html <- StnChkCoordsHtml()
    html <- StnChkCoordsHtmlParse(html, to_repl)

    return(html)
}

StnChkCoordsBrowse <- function(html.page){
    html.file <- "StationsCoordinates.html"
    tmpdir <- tempdir()
    tmpfile <- file.path(tmpdir, html.file)
    cat(html.page, file = tmpfile, sep = "\n")

    markers <- c('marker-shadow.png', 'marker-icon-blue.png',
                 'marker-icon-orange.png', 'marker-icon-red.png',
                 'marker-icon-green.png')
    file.copy(file.path(.cdtDir$Root, 'images', markers),
              file.path(tmpdir, markers),
              overwrite = TRUE)

    html.url <- paste0(.cdtEnv$cdtUrl, html.file)
    if(interactive()){
        utils::browseURL(html.url)
    }else{
        return(NULL)
    }
    invisible()
}
