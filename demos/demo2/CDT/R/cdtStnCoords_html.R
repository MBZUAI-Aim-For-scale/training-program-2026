StnChkCoordsHtmlParse <- function(html, replace_text){
    for(j in seq_along(replace_text)){
        txt <- names(replace_text[j])
        pattern <- paste0("<<<< *", txt, " *>>>>")
        html <- gsub(pattern, replace_text[[j]], html)
    }

    return(html)
}

StnChkCoordsHtml <- function(){
tmp <- '<!DOCTYPE html>
<html>

<head>
    <title> Stations coordinates </title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin="" />
    <script src="https://cdn.jsdelivr.net/npm/jquery@3.7.1/dist/jquery.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
    <!-- <script src="https://maps.googleapis.com/maps/api/js?key=\'<<<< googleAPIKey >>>>\'" async defer></script> -->
    <style>
    html,
    body {
        height: 100%;
    }

    #mapAWSCoords {
        width: 100%;
        height: 95%;
        border: 1px solid #AAA;
    }
    #maptype{
        width: 200px;
    }
    .leaflet-attribution-flag {
        display: none !important;
    }
    </style>
</head>

<body>
    <div>
        <span>
            <label>Change map style:</label>
            <select name="maptype" id="maptype" class="maptype">
                <option value="openstreetmap" selected>OpenStreetMap Standard</option>
                <option value="mapboxsatellitestreets">Mapbox Satellite Streets</option>
                <option value="mapboxsatellite">Mapbox Satellite</option>
                <option value="mapboxstreets">Mapbox Streets</option>
                <option value="mapboxoutdoors">Mapbox Outdoors</option>
                <option value="mapboxlight">Mapbox Light</option>
                <option value="googlemaps">Google Maps</option>
            </select>
        </span>
    </div>
    <div id="mapAWSCoords"></div>
    <script>
    var serverPath = "<<<< serverPath >>>>";
    $(document).ready(function() {
        var mymap = L.map("mapAWSCoords", {
            center: [<<<<lat_c>>>>, <<<< lon_c>>>>],
            minZoom: 2,
            zoom: 8
        });
        var mytile = "";
        mytile = L.tileLayer("http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: \'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>\',
            maxZoom: 19,
            subdomains: ["a", "b", "c"]
        }).addTo(mymap);
        var iconshadow = serverPath + "marker-shadow.png";
        var blueIcon = new L.Icon({
            iconUrl: serverPath + "marker-icon-blue.png",
            shadowUrl: iconshadow,
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        });
        var orangeIcon = new L.Icon({
            iconUrl: serverPath + "marker-icon-orange.png",
            shadowUrl: iconshadow,
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        });
        var redIcon = new L.Icon({
            iconUrl: serverPath + "marker-icon-red.png",
            shadowUrl: iconshadow,
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        });
        var greenIcon = new L.Icon({
            iconUrl: serverPath + "marker-icon-green.png",
            shadowUrl: iconshadow,
            iconSize: [25, 41],
            iconAnchor: [12, 41],
            popupAnchor: [1, -34],
            shadowSize: [41, 41]
        });
        var icons = {
            blue: {
                icon: blueIcon
            },
            orange: {
                icon: orangeIcon
            },
            red: {
                icon: redIcon
            }
        };
        $.getJSON(serverPath + "station_coords.json", function(json) {
            $.each(json, function() {
                var contenu = <<<< contenu >>>>;
                if (typeof this.LonX !== "undefined") {
                    L.marker([this.LatX, this.LonX], { icon: icons[this.StatusX].icon }).bindPopup(contenu).addTo(mymap);
                }
            });
        });
        $("#maptype").on("change", function() {
            mymap.removeLayer(mytile);
            mymap.attributionControl.removeAttribution();
            var maptype = $("#maptype option:selected").val();
            if (maptype == "openstreetmap") {
                mytile = L.tileLayer("http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                    attribution: \'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>\',
                    maxZoom: 19,
                    subdomains: ["a", "b", "c"]
                }).addTo(mymap);
            } else if (maptype == "googlemaps") {
                mytile = L.tileLayer("http://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", {
                    attribution: "&copy; Google Maps",
                    maxZoom: 20,
                    subdomains: ["mt0", "mt1", "mt2", "mt3"]
                }).addTo(mymap);
            } else {
                var mapid = "";
                if (maptype == "mapboxsatellite") {
                    mapid = "satellite-v9";
                } else if (maptype == "mapboxstreets") {
                    mapid = "streets-v11";
                } else if (maptype == "mapboxoutdoors") {
                    mapid = "outdoors-v11";
                } else if (maptype == "mapboxlight") {
                    mapid = "light-v10";
                } else {
                    mapid = "satellite-streets-v11";
                }
                mytile = L.tileLayer("https://api.mapbox.com/styles/v1/mapbox/{id}/tiles/{z}/{x}/{y}?access_token={accessToken}", {
                    attribution: \'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery &copy; <a href="https://www.mapbox.com/">Mapbox</a>\',
                    tileSize: 512,
                    zoomOffset: -1,
                    maxZoom: 23,
                    id: mapid,
                    accessToken: "pk.eyJ1IjoicmlqYWYiLCJhIjoiY2xoOGJ5djZvMDc1NTNlcXk5bmRyMzVuMSJ9.WtjNPGY0JwLwbkacp8H8UQ"
                }).addTo(mymap);
            }
            window.mytile = mytile;
        });
        var marker1;
        mymap.on("dblclick", function(e) {
            if (!marker1) {
                marker1 = L.marker(e.latlng, { icon: greenIcon }).addTo(mymap);
            } else {
                marker1.setLatLng(e.latlng);
            }
            var position = marker1.getLatLng();
            marker1.bindPopup("<b>Latitude : </b>" + position.lat + "<br />" + "<b>Longitude : </b>" + position.lng).openPopup();
        });
    });
    </script>
</body>

</html>
'
	file <- file.path(tempdir(), 'tmp.StnChkCoords.html')
	cat(tmp, file = file)
	tmp <- readLines(file)
	unlink(file)

	return(tmp)
}

