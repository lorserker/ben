<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="robots" content="noindex, nofollow">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <title>BEN Autoplay - Generate PBN</title>
    <style>
        /* Reset any external styles */
        .autoplay-container,
        .autoplay-container * {
            box-sizing: border-box;
        }
        .autoplay-container .container,
        .autoplay-container .content {
            display: block !important;
            float: none !important;
            width: auto !important;
            max-width: none !important;
        }
        body {
            background: #f5f6f8;
            margin: 0;
            padding: 0;
        }
        .autoplay-container {
            max-width: 700px;
            margin: 30px auto;
            padding: 0 20px;
        }
        .autoplay-container .card {
            background: #fff;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            padding: 28px 32px;
            margin-bottom: 20px;
            display: block !important;
            float: none !important;
        }
        .autoplay-container .card-header {
            margin-bottom: 24px;
            text-align: center;
        }
        .autoplay-container .card-header h2 {
            margin: 0 0 8px 0;
            color: #2c3e50;
            font-size: 1.5em;
            font-weight: 600;
        }
        .autoplay-container .card-header p {
            margin: 0;
            color: #7f8c8d;
            font-size: 0.95em;
        }
        .autoplay-container .form-row {
            margin-bottom: 20px;
            display: block;
        }
        .autoplay-container .form-row label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            color: #34495e;
            font-size: 0.9em;
        }
        .autoplay-container .form-row input[type="text"],
        .autoplay-container .form-row input[type="number"] {
            width: 100%;
            padding: 12px 14px;
            font-size: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            transition: border-color 0.2s, box-shadow 0.2s;
            box-sizing: border-box;
        }
        .autoplay-container .form-row input:focus {
            outline: none;
            border-color: #3498db;
            box-shadow: 0 0 0 3px rgba(52,152,219,0.15);
        }
        .autoplay-container .form-row input[type="number"] {
            width: 100px;
        }
        .autoplay-container .input-hint {
            display: block;
            margin-top: 6px;
            font-size: 0.85em;
            color: #95a5a6;
        }
        .autoplay-container .form-row-inline {
            display: flex;
            gap: 20px;
        }
        .autoplay-container .form-row-inline .form-row {
            flex: 1;
        }
        .autoplay-container .form-row-inline .form-row:first-child {
            flex: 2;
        }
        .autoplay-container .button-row {
            display: flex;
            gap: 12px;
            margin-top: 28px;
            padding-top: 24px;
            border-top: 1px solid #ecf0f1;
        }
        .autoplay-container .btn {
            padding: 14px 28px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            border: none;
            border-radius: 8px;
            transition: all 0.2s;
        }
        .autoplay-container .btn:active {
            transform: scale(0.98);
        }
        .autoplay-container .btn-primary {
            background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
            color: white;
            flex: 1;
        }
        .autoplay-container .btn-primary:hover {
            background: linear-gradient(135deg, #2980b9 0%, #1f6dad 100%);
            box-shadow: 0 4px 12px rgba(52,152,219,0.3);
        }
        .autoplay-container .btn-primary:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
            box-shadow: none;
        }
        .autoplay-container .btn-secondary {
            background: #ecf0f1;
            color: #7f8c8d;
        }
        .autoplay-container .btn-secondary:hover {
            background: #e0e0e0;
            color: #555;
        }
        .autoplay-container #result-section {
            margin-top: 24px;
            display: block !important;
        }
        .autoplay-container .result-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 18px;
            display: block !important;
            float: none !important;
            width: 100% !important;
            clear: both;
        }
        .autoplay-container .result-card h3 {
            margin: 0 0 14px 0;
            font-size: 0.85em;
            color: #7f8c8d;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }
        .autoplay-container #result-summary {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            white-space: pre-wrap;
            line-height: 1.6;
            color: #2c3e50;
            display: block;
        }
        .autoplay-container #pbn-output {
            width: 100% !important;
            height: 280px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            padding: 14px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background: #fff;
            resize: vertical;
            box-sizing: border-box;
            display: block;
        }
        .autoplay-container #pbn-output:focus {
            outline: none;
            border-color: #3498db;
        }
        .autoplay-container .pbn-actions {
            display: flex;
            gap: 10px;
            margin-top: 14px;
        }
        .autoplay-container .pbn-actions .btn {
            padding: 10px 18px;
            font-size: 13px;
        }
        .autoplay-container .status-message {
            padding: 14px 18px;
            margin: 20px 0;
            border-radius: 8px;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .autoplay-container .status-message.info {
            background: #ebf5fb;
            border: 1px solid #aed6f1;
            color: #2471a3;
        }
        .autoplay-container .status-message.info::before {
            content: "...";
            animation: dots 1.5s infinite;
        }
        @keyframes dots {
            0%, 20% { content: "."; }
            40% { content: ".."; }
            60%, 100% { content: "..."; }
        }
        .autoplay-container .status-message.error {
            background: #fdedec;
            border: 1px solid #f5b7b1;
            color: #c0392b;
        }
        .autoplay-container .status-message.success {
            background: #e9f7ef;
            border: 1px solid #a9dfbf;
            color: #27ae60;
        }
        .autoplay-container .back-link {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            color: #3498db;
            text-decoration: none;
            font-size: 14px;
            font-weight: 500;
            transition: color 0.2s;
        }
        .autoplay-container .back-link:hover {
            color: #2980b9;
        }
        .autoplay-container .page-title {
            text-align: center;
            margin-bottom: 24px;
        }
        .autoplay-container .page-title h1 {
            margin: 0;
            color: #2c3e50;
            font-size: 1.8em;
            font-weight: 700;
        }
    </style>
</head>
<body>
<div class="autoplay-container">
    <div class="page-title">
        <h1>BEN Autoplay</h1>
    </div>

    <div class="card">
        <div class="card-header">
            <h2>Generate PBN</h2>
            <p>Play a complete board with all 4 hands controlled by BEN</p>
        </div>

        <div class="form-row-inline">
            <div class="form-row">
                <label for="api-url">API URL</label>
                <input type="text" id="api-url" value="http://localhost:8085">
            </div>
            <div class="form-row">
                <label for="board">Board #</label>
                <input type="number" id="board" value="1" min="1" max="16">
            </div>
        </div>

        <div class="form-row">
            <label for="deal">Deal (required)</label>
            <input type="text" id="deal" placeholder="862.62.AQT52.A96 AQJT9.Q875.97.K7 7543.AT943.8.JT8 K.KJ.KJ643.Q5432">
            <span class="input-hint">Format: N E S W hands separated by spaces</span>
        </div>

        <div class="button-row">
            <button id="run-btn" class="btn btn-primary" onclick="runAutoplay()">Run Autoplay</button>
            <button class="btn btn-secondary" onclick="clearResults()">Clear</button>
        </div>

        <div id="status"></div>

        <div id="result-section" style="display: none;">
            <div class="result-card">
                <h3>Result Summary</h3>
                <div id="result-summary"></div>
            </div>

            <div class="result-card">
                <h3>PBN Output</h3>
                <textarea id="pbn-output" readonly></textarea>
                <div class="pbn-actions">
                    <button class="btn btn-secondary" onclick="copyPbn()">Copy to Clipboard</button>
                    <button class="btn btn-secondary" onclick="downloadPbn()">Download PBN</button>
                </div>
            </div>
        </div>
    </div>

    <a href="/home" class="back-link">← Back to Home</a>
</div>

<script>
    function setStatus(message, type) {
        const statusDiv = document.getElementById('status');
        statusDiv.innerHTML = `<div class="status-message ${type}">${message}</div>`;
    }

    function clearResults() {
        document.getElementById('result-section').style.display = 'none';
        document.getElementById('result-summary').innerHTML = '';
        document.getElementById('pbn-output').value = '';
        document.getElementById('status').innerHTML = '';
    }

    async function runAutoplay() {
        const apiUrl = document.getElementById('api-url').value.trim();
        const board = document.getElementById('board').value;
        const deal = document.getElementById('deal').value.trim();

        if (!deal) {
            setStatus('Error: Deal is required', 'error');
            return;
        }

        const runBtn = document.getElementById('run-btn');
        runBtn.disabled = true;

        setStatus('Running autoplay... (this may take 30-60 seconds)', 'info');

        try {
            let url = `${apiUrl}/autoplay?board=${board}&deal=${encodeURIComponent(deal)}`;

            const response = await fetch(url);
            const data = await response.json();

            if (data.error) {
                setStatus(`Error: ${data.error}`, 'error');
                runBtn.disabled = false;
                return;
            }

            // Display result summary
            const auction = Array.isArray(data.auction) ? data.auction.join('-') : data.auction;
            const play = Array.isArray(data.play) ? data.play.join(' ') : (data.play || 'N/A');

            let summary = `Board: ${data.boardNumber || board}
Dealer: ${data.dealer}
Vulnerability: ${data.vulnerability}
Deal: ${data.deal}

Auction: ${auction}
Contract: ${data.contract}
Declarer: ${data.declarer || 'N/A'}
Tricks: ${data.tricks}
Score: ${data.score} (NS: ${data.ns_score || data.nsScore})

Play: ${play}
Elapsed: ${data.elapsed || 'N/A'} seconds`;

            document.getElementById('result-summary').textContent = summary;

            // Generate PBN
            const pbn = generatePbn(data, board);
            document.getElementById('pbn-output').value = pbn;

            document.getElementById('result-section').style.display = 'block';
            setStatus('Autoplay completed successfully!', 'success');

        } catch (error) {
            setStatus(`Error: ${error.message}`, 'error');
        }

        runBtn.disabled = false;
    }

    function generatePbn(data, boardNum) {
        const auction = Array.isArray(data.auction) ? data.auction : (data.auction || '').split('-');
        const play = Array.isArray(data.play) ? data.play : [];

        let pbn = `[Event "BEN Autoplay"]
[Site ""]
[Date "${new Date().toISOString().split('T')[0].replace(/-/g, '.')}"]
[Board "${boardNum}"]
[West "BEN"]
[North "BEN"]
[East "BEN"]
[South "BEN"]
[Dealer "${data.dealer}"]
[Vulnerable "${formatVuln(data.vulnerability)}"]
[Deal "N:${data.deal}"]
[Declarer "${data.declarer || ''}"]
[Contract "${data.contract}"]
[Result "${data.tricks}"]
`;

        // Add auction
        pbn += `[Auction "${data.dealer}"]\n`;
        let auctionLine = '';
        let bidCount = 0;
        for (let i = 0; i < auction.length; i++) {
            let bid = auction[i];
            if (bid === 'PAD_START') continue;
            if (bid === 'PASS') bid = 'Pass';
            auctionLine += bid + ' ';
            bidCount++;
            if (bidCount % 4 === 0) {
                pbn += auctionLine.trim() + '\n';
                auctionLine = '';
            }
        }
        if (auctionLine.trim()) {
            pbn += auctionLine.trim() + '\n';
        }

        // Add play if available
        if (play.length > 0) {
            const openingLeader = getOpeningLeader(data.dealer, data.declarer);
            pbn += `[Play "${openingLeader}"]\n`;
            let playLine = '';
            for (let i = 0; i < play.length; i++) {
                playLine += play[i] + ' ';
                if ((i + 1) % 4 === 0) {
                    pbn += playLine.trim() + '\n';
                    playLine = '';
                }
            }
            if (playLine.trim()) {
                pbn += playLine.trim() + '\n';
            }
        }

        return pbn;
    }

    function formatVuln(vuln) {
        if (!vuln) return 'None';
        const v = vuln.toUpperCase();
        if (v === 'NONE' || v === 'LOVE' || v === '-') return 'None';
        if (v === 'NS' || v === 'N-S') return 'NS';
        if (v === 'EW' || v === 'E-W') return 'EW';
        if (v === 'BOTH' || v === 'ALL') return 'All';
        return vuln;
    }

    function getOpeningLeader(dealer, declarer) {
        const positions = ['N', 'E', 'S', 'W'];
        const decl_i = positions.indexOf(declarer);
        if (decl_i === -1) return 'N';
        return positions[(decl_i + 1) % 4];
    }

    function copyPbn() {
        const pbnOutput = document.getElementById('pbn-output');
        pbnOutput.select();
        document.execCommand('copy');
        alert('PBN copied to clipboard!');
    }

    function downloadPbn() {
        const pbn = document.getElementById('pbn-output').value;
        const board = document.getElementById('board').value;

        const blob = new Blob([pbn], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `ben_autoplay_board${board}.pbn`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }
</script>
</body>
</html>
