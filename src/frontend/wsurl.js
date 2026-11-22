// Addresses of the BEN servers as the browser must reach them.
//
// Game server (gameserver.py):
//
// Opened directly (http://host:8080), the page talks to the game server on
// that server's own port: ws://host:4443/.
//
// Behind a reverse proxy or tunnel the page arrives on the default port and
// the game server ports are not reachable from outside. The page then uses a
// path on its own origin, wss://host/ws/4443/, and the proxy routes
// /ws/<port>/ to that port. gameserver.py reads only the query string of the
// request, so it accepts the path as it is.
//
// A proxy can also route by port, as on ben.aalborgdata.dk: the page is on the
// default port, but the game servers and the API keep their own ports
// (wss://host:4443/, https://host:8085). appserver.py --direct-ports serves
// this file with BEN_DIRECT_PORTS = true in front for that case.
var BEN_DIRECT_PORTS = (typeof BEN_DIRECT_PORTS !== 'undefined') && BEN_DIRECT_PORTS === true;

function benBehindPathProxy() {
    return location.port === '' && !BEN_DIRECT_PORTS;
}

function benGameServerUrl(serverPort) {
    var base = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.hostname;
    if (benBehindPathProxy()) {
        return base + '/ws/' + serverPort + '/';
    }
    return base + ':' + serverPort + '/';
}

// REST API (gameapi.py), no trailing slash: http://host:8085 when the page is
// opened directly, https://host/gameapi/8085 behind a proxy. gameapi.py drops
// the /gameapi/<port> prefix from the request path.
function benApiUrl(apiPort) {
    apiPort = apiPort || 8085;
    if (benBehindPathProxy()) {
        return location.protocol + '//' + location.hostname + '/gameapi/' + apiPort;
    }
    return location.protocol + '//' + location.hostname + ':' + apiPort;
}
