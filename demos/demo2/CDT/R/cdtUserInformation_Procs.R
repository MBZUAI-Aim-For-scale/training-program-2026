
cdtUserInformation <- function(){
    userInfoFile <- file.path(.cdtDir$dirLocal, 'config', 'user_infos')
    if(!file.exists(userInfoFile)){
        userForms <- cdtUserInfo()
        if(!is.null(userForms)){
            userForms$created <- format(as.POSIXct(Sys.time()), '%Y%m%d%H%M%S', tz = 'UTC')
            ## Saved now, before sending: the send below runs in a background future that
            ## on.exit() stops as soon as this function returns, so it often never got to
            ## save the file, and the form came back on every start. (Training program fix.)
            userForms$sent <- FALSE
            saveRDS(userForms, userInfoFile)
            sendForms <- testConnection()
        }else{
            sendForms <- FALSE
        }
    }else{
        userForms <- readRDS(userInfoFile)
        if(userForms$sent){
            sendForms <- FALSE
        }else{
            if(testConnection()){
                sendForms <- TRUE
            }else{
                sendForms <- FALSE
            }
        }
    }

    if(sendForms){
        future::plan("multisession")
        on.exit(future::plan("sequential"))
        future::future({
            cdt_api <- readRDS(file.path(.cdtDir$Root, 'data', 'tmp_url'))
            res <- httr::POST(cdt_api$endpoint, body = userForms, encode = "json")
            if(httr::status_code(res) == 200){
                ret <- httr::content(res)
                if(ret$response == 'ok'){
                    userForms$sent <- TRUE
                }else{
                    userForms$sent <- FALSE
                }
            }else{
                userForms$sent <- FALSE
            }
            saveRDS(userForms, userInfoFile)
        })
    }

    invisible()
}
