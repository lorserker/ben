<!DOCTYPE html>

<head>
	<title>BBA Bidding</title>

	<!-- CSS -->
	<style>
        body { 
            width: 1000px; /* Set a fixed width for the body element */ 
            margin: 0 auto; /* Center the body element within its parent */ 
        } 
		table {
			border-collapse: collapse;
			width: 800px;
		}

		th,
		td {
			text-align: left;
			padding: 3px;
		}

		tr.us:nth-child(even) {
			background-color: #d4e9d4;
		}
        tr.them:nth-child(even) {
			background-color: #f5f5dc;
		}

		th {
			background-color: #7c0f65;
			color: white;
		}

		td:first-child {
 		   text-align: center; /* Center the content of the first cell in each row */
		}
        /* Basic styles for the bidding diagram */
        .biddingdiagram {
			width: 400px;
            border-collapse: collapse;
            margin: 10px;
            margin-left: 200px;
        }

        .biddingdiagram th, .biddingdiagram td {
            border: 1px solid #000;
			padding: 3px;
            text-align: center;
        }

		.spades { color: #0000ff; }
        .hearts { color: #ff0000; }
        .diamonds { color: #ffc000; }
        .clubs { color: #008000; }

	</style>

	<!--JavaScript-->
	<script>
		let biddingsequence = getQueryParam('bid')
		if (!biddingsequence) {
			biddingsequence = "*"
		} else if (biddingsequence.endsWith("-")) {
			biddingsequence += "*";
		}

		function create_URL() {

			// Construct the URL dynamically
			var url = "http://" + window.location.hostname + ":8085/bids";
			var params = [];

			if (fileUs) params.push("file_us=" + encodeURIComponent(fileUs));
			if (fileThem) params.push("file_them=" + encodeURIComponent(fileThem));
			if (biddingsequence) params.push("ctx=" + encodeURIComponent(biddingsequence));

			if (params.length > 0) {
				url += "?" + params.join("&"); // Append only if there are parameters
			}
			return url
		}

        function getQueryParam(key) {
            const queryString = window.location.search;
            const searchParams = new URLSearchParams(queryString);
            return searchParams.get(key);
        }

		function loadJSONDoc() {
			let cachedData = localStorage.getItem('bbaData_' + biddingsequence);

			if (cachedData) {
				// If there's cached data, parse it as JSON
				let jsonData = JSON.parse(cachedData);
				displayBidDetails(jsonData); // Use cached data
			} else {
				let xmlhttp = new XMLHttpRequest();
				xmlhttp.onreadystatechange = function () {
					if (this.readyState == 4 && this.status == 200) {
						let data = this.responseText;
						localStorage.setItem('bbaData_' + biddingsequence, data); // Cache the response
						let jsonData = JSON.parse(data); // Parse the JSON response
						displayBidDetails(jsonData);
					}
				};
				xmlhttp.open("GET", url, true);
				xmlhttp.send();
			}
		}

		function biddingended(bid) {
			return (biddingsequence.endsWith("-P-P-*") || biddingsequence.endsWith("-P-P-P"))  && bid == "P"
		}

		function canbid(bid) {
			return !biddingsequence.endsWith("-P-P-P")
		}

		function displayBidDetails(jsonData) {
			const numOfHyphens = (biddingsequence.match(/-/g) || []).length;
			const who = numOfHyphens % 2 === 0 ? 'us' : 'them';
			let table = `<tr><th>Bid</th><th>Description</th></tr>`;

			// Assuming jsonData contains an array of bids or a similar structure
			jsonData.forEach(bid => {
				const bValue = bid.bid.toUpperCase();
				let bValueHtml = bValue;
				if (bValue != "D") {
					bValueHtml = bValue.replace(/S/g, '<span style="color: blue">&spades;</span>');
					bValueHtml = bValueHtml.replace(/H/g, '<span style="color: red">&hearts;</span>');
					bValueHtml = bValueHtml.replace(/D/g, '<span style="color: orange">&diams;</span>');
					bValueHtml = bValueHtml.replace(/C/g, '<span style="color: green">&clubs;</span>');
				} else {
					bValueHtml = bValue;
				}
				let	mValueHtml = "&nbsp";
				if (!bid.m.includes("Not defined") && !bid.m.includes("unknown")) {
					mValueHtml = bid.m.replace(/!S/g, '<span style="color: blue">&spades;</span>');
					mValueHtml = mValueHtml.replace(/!H/g, '<span style="color: red">&hearts;</span>');
					mValueHtml = mValueHtml.replace(/!D/g, '<span style="color: orange">&diams;</span>');
					mValueHtml = mValueHtml.replace(/!C/g, '<span style="color: green">&clubs;</span>');
					mValueHtml = replaceSuitsBeforeDoubleDash(mValueHtml);
					mValueHtml = replaceSuits(mValueHtml);
				}				

				if (!canbid(bValueHtml)) {
					table += `<tr class="${who}"><td></td><td>Bidding ended</td></tr>`;
				} else {
					table += `<tr class="${who}"><td>`
					if (biddingended(bValueHtml))
						table += `P`
					else {
						table += `<a href="?bid=`
						table += `${biddingsequence.substring(0, biddingsequence.length - 1)}`
					   	table += `${bValue}-*">${bValueHtml}</a>`
					} 
					table += `</td><td>${mValueHtml}</td></tr>`;
				}
			});

			document.getElementById("id").innerHTML = table;
		}

		// Define a mapping of card suits to HTML symbols
		const suitMapping = {
			'C': '<span style="color: green">&clubs;</span>', 
			'D': '<span style="color: orange">&diams;</span>', 
			'H': '<span style="color: red">&hearts;</span>',
			'S': '<span style="color: blue">&spades;</span>'
		};

		// Define a function to replace card codes with HTML symbols
		function replaceSuits(text) {
			// Create a regular expression pattern to match card codes
			const pattern = /\b[1-7][CDHS]\b/g;
			
			// Replace function that will use the suitMapping to replace suits
			return text.replace(pattern, (match) => {
				const suit = match[1]; // Extract the suit character (C, D, H, S)
				return match[0] + suitMapping[suit] || match; // Replace with the HTML symbol or leave unchanged
			});
		}

		// Define a function to replace suit symbols only before --
		function replaceSuitsBeforeDoubleDash(text) {

			if (!text.includes('--')) {
    		    return text; // No replacement needed if '--' is not present
    		}
				// Split the text at '--'
			const [beforeDoubleDash, afterDoubleDash] = text.split('--', 2);
			
			// Create a regular expression pattern to match suit symbols alone
			const pattern = /(?<=\s|\(|^)[1-7]?[CDHS](?=\s|\)|$)/g;

			// Replace suits in the part before '--'
			const replacedBeforeDoubleDash = beforeDoubleDash.replace(pattern, (match) => {
				// Extract the suit character (C, D, H, S) based on its position
				const suit = match.match(/[CDHS]/)[0]; // Match the suit character
				return suitMapping[suit] || match; // Replace with the HTML symbol or leave unchanged
			});			
			// Combine the parts back together
			return replacedBeforeDoubleDash + (afterDoubleDash ? '--' + afterDoubleDash : '');
		}		
	</script>
</head>

<body>

	<br><br>
	<h1>BBA Bidding Definitions</h1>
	<h2>Current bidding: </h2>
    <table class="biddingdiagram">
		<tbody id="biddingTable">
            <tr>
                <th width="25%">We</th>
                <th width="25%">They</th>
                <th width="25%">We</th>
                <th width="25%">They</th>
            </tr>
        </tbody>
    </table>
	<br>
        <label for="file_us">Select BBA for Us:</label>
        <select name="file_us" id="file_us">
			<option value="">-- Select a CC --</option>  <!-- Default blank option -->
            % for display_name, full_name in file_map.items():
                <option value="{{ full_name }}">{{ display_name }}</option>
            % end
        </select>

        <label for="file_them">Select BBA for Them:</label>
        <select name="file_them" id="file_them">
			<option value="">-- Select a CC --</option>  <!-- Default blank option -->
            % for display_name, full_name in file_map.items():
                <option value="{{ full_name }}">{{ display_name }}</option>
            % end
        </select>

    	<button id="setFileBtn">Set CC's</button>  <!-- Saves selection -->

        <br>
			<p id="statusMessage" style="color: green;"></p>  <!-- Status message -->
		<br>
    <script>
        function formatBid(bid) {
			if (bid === "D") {
        		return "X";
    		}
    
	
			return bid.replace(/S/g, '<span class="spades">&spades;</span>')
						.replace(/H/g, '<span class="hearts">&hearts;</span>')
						.replace(/D/g, '<span class="diamonds">&diams;</span>')
						.replace(/C/g, '<span class="clubs">&clubs;</span>')
						.replace(/R/g, 'XX')
						.replace(/\*/g, '?');
        }

		// Get the query string from the URL
        const queryString = window.location.search;

        // Create a URLSearchParams object
        const urlParams = new URLSearchParams(queryString);

        // Get the value of the 'bid' parameter
        let bidValue = urlParams.get('bid');

        // Replace the last '*' with '?'
        if (bidValue) {

			const bids = bidValue.split('-');

            // Select the table body
            const biddingTable = document.getElementById('biddingTable');

            // Create rows dynamically
            for (let i = 0; i < bids.length; i += 4) {
                // Create a new row
                const row = document.createElement('tr');

                // Create and populate cells for this row
                for (let j = 0; j < 4; j++) {
                    const cell = document.createElement('td');
                    if (bids[i + j]) {
                        cell.innerHTML = formatBid(bids[i + j]);
                    } else {
                        cell.innerHTML = '&nbsp;';
                    }
                    row.appendChild(cell);
                }

                // Append the row to the table
                biddingTable.appendChild(row);
			}
		}
		    // Function to clear cache
		function clearCache() {
            var fileUs = localStorage.getItem("file_us");
            var fileThem = localStorage.getItem("file_them");
			// Clear all localStorage
			localStorage.clear(); 
            if (fileUs) localStorage.setItem("file_us", fileUs);
            if (fileThem) localStorage.setItem("file_them", fileThem);
					
			alert('Cache has been cleared!');
			// Reload the page after clearing cache
    		location.reload(true);
		}
		</script>
	<table id="id"></table>
	<h2>Trouble shooting</h2>
		<p>The data is copied to your browser's local storage. If you see strange result, try clearing the local data, and you can see where the data come from by clicking on the link below.</p>
		<a id="sourceLink" href="" target="_blank">Visit Source</a>
		<!-- Clear Cache Button -->
		<button id="clearCacheButton">Clear Cache</button>
	</h2>
    <script>
        // Save the selected file values
        document.getElementById("setFileBtn").addEventListener("click", function () {
            var fileUs = document.getElementById("file_us").value;
            var fileThem = document.getElementById("file_them").value;

            if (fileUs) localStorage.setItem("file_us", fileUs);
            if (fileThem) localStorage.setItem("file_them", fileThem);

            document.getElementById("statusMessage").innerText = "CC set successfully!";
			clearCache()
        });
		
    	// Attach the clearCache function to the button
    	document.getElementById('clearCacheButton').onclick = clearCache;
        // Restore saved values from localStorage
        window.onload = function () {
            fileUs = localStorage.getItem("file_us");
            fileThem = localStorage.getItem("file_them");

            if (fileUs) {
                document.getElementById("file_us").value = fileUs;
            }
            if (fileThem) {
                document.getElementById("file_them").value = fileThem;
            }
			url = create_URL()
			// Set the href attribute of the link
			document.getElementById('sourceLink').href = url;
			loadJSONDoc()
        };
 	</script>
</body>

</html>
