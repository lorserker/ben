// Address of a BEN game server (gameserver.py) as the browser must reach it.
//
// Opened directly (http://host:8080), the page talks to the game server on
// that server's own port: ws://host:4443/.
//
// Behind a reverse proxy or tunnel the page arrives on the default port and
// the game server ports are not reachable from outside. The page then uses a
// path on its own origin, wss://host/ws/4443/, and the proxy routes
// /ws/<port>/ to that port. gameserver.py reads only the query string of the
// request, so it accepts the path as it is.
function benGameServerUrl(serverPort) {
    var base = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.hostname;
    if (location.port === '') {
        return base + '/ws/' + serverPort + '/';
    }
    return base + ':' + serverPort + '/';
}
