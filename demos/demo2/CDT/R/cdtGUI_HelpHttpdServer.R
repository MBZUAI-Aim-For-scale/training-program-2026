
createProxyPass <- function(path, query, ...){
    path <- gsub("^/custom/CDT/", "", path)
    tmpfile <- sprintf("%s%s%s", tempdir(), .Platform$file.sep, path)
    list(file = tmpfile,
         "content-type" = "text/html",
         "status code" = 200L)
}

startHelpHttpdServer <- function(){
    options(help_type = "html")
    if(!is.HelpServerRunning()) tools::startDynamicHelp()
    env <- get(".httpd.handlers.env", asNamespace("tools"))
    env[["CDT"]] <- createProxyPass
    port <- ifelse(R.version['svn rev'] < 67550 | getRversion() < "3.2.0", 
                   get("httpdPort", envir = environment(tools::startDynamicHelp)),
                   tools::startDynamicHelp(NA))
    .cdtEnv$cdtUrl <- sprintf('http://127.0.0.1:%s/custom/CDT/', port)
}
