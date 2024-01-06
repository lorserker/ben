<html> 
	<head> 
    <meta charset="utf-8">
		<title>BEN - The open source bridge engine</title> 
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <link rel="stylesheet" href="/app/style.css">
	</head> 
    <script type="text/javascript">
        function copyToClipboard(idx) {
        const bbaText = document.getElementById('bbaText'+idx);
        const text = bbaText.textContent;

        const el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);

        // You can add a message or any other functionality after copying
        alert('Copied to clipboard: ' + text);
        }
    </script>
<body> 

<center><h1>Play with BEN</a></h1></center>

<div class="container">
  <h2>Play this deal: </h2>

  <div class="content">
    <div>
    <form action="/submit" method="post">
        <label for="board">Board:</label>
        <input type="input" id="board" name="board"><br>
        <label for="dealer">Dealer:</label>
        <select id="dealer" name="dealer">
            <option value="N">North</option>
            <option value="S">South</option>
            <option value="E">East</option>
            <option value="W">West</option>
        </select><br>
        <label for="vulnerable">Vulnerability:</label>
        <select id="vulnerable" name="vulnerable">
            <option value="None">None</option>
            <option value="NS">NS</option>
            <option value="EW">EW</option>
            <option value="Both">Both</option>
        </select><br>
        <label for="deal">Text:</label>
        <textarea id="deal" name="dealtext" cols="40"></textarea><br>
        <input type="submit" value="Play from text">
    </form>
    <form action="/submit" method="post">
        <label for="deal">PBN:</label>
        <textarea id="deal" name="dealpbn" cols="40"  rows="10"></textarea><br>
        <input type="submit" value="Play from PBN">
    </form>
    </div>
    <div>
    <form action="/submit" method="post">
        <label for="deal">LIN:</label>
        <textarea id="deallin" name="deallin" cols="40" rows="3"></textarea><br>
        <input type="submit" value="Play from LIN">
    </form>
    <br>
    <br>
    <br>
    </div>
    <div>
    <form action="/submit" method="post">
        <label for="deal">BBA:</label>
        <textarea id="dealbba" name="dealbba" cols="40"></textarea><br>
        <input type="submit" value="Play from BBA">
    <br>
    <br>
    <br>
    </form>
    </div>
    <div>
    <form action="/submit" method="post">
        <label for="board">Board:</label>
        <input type="input" id="board" name="board"><br>
        <input type="submit" value="Play random">
    <br>
    <br>
    <br>
    </form>
    </div>
</div>

<div class="container">
  <h2>Previous played deals</h2>

  <div class="content">
<ul>
% for index, deal in enumerate(deals):
    <li>
        <span>{{deal['board_no_index']}}  <a href="/app/viz.html?deal={{deal['deal_id']}}{{deal['board_no_ref']}}">{{deal['contract']}}{{deal.get('trick_winners_count', '')}}</a></span>&nbsp;&nbsp;
        <span><a href="{{deal['delete_url']}}">delete</a></span><br>
        <span class="bba">BBA=<span id="bbaText{{index}}">{{deal['bba']}}&nbsp;<i class="fas fa-copy" onclick="copyToClipboard({{index}})"></i>
        </span>
        </span>
    </li>
% end
</ul>
</div>
</div>

</body> 
</html> 
