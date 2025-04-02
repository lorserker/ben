function generateSamplesTable(samples, id) {
    if (!samples || samples.length === 0) {
        return '';
    }

    tableId = "sampleLinesPlay" + id
    // Determine number of probability columns from the first sample
    const firstParts = samples[0].trim().split(" - ");
    const hasBidding = firstParts[1]?.includes("|");
    const probabilityData = hasBidding ? firstParts[1].split("|")[0].trim() : firstParts[1];
    const probabilityCount = probabilityData?.split(/\s+/).length || 0;

    // Static headers and corresponding labels
    const staticHeaders = ["Deal"];
    const probabilityLabels = ["Bid Score", "Deal Prob", "Lead Prob", "Play Score", "Logic Score", "Discard Score"];

    // Construct table headers dynamically
    const headers = staticHeaders.map(h => `<th>${h}</th>`).join('') +
                    probabilityLabels.slice(0, probabilityCount).map(h => `<th class="right-align">${h}</th>`).join('');

    // Generate table rows
    const rows = samples.map(sample => {
        const parts = sample.split(" - ");
        const deal = parts[0];

        // Handle multiple biddings
        let biddingLines = [];
        let probabilities = [];
        if (parts[1]?.includes("|")) {
            const splitParts = parts[1].split("|");
            probabilities = splitParts[0].trim().split(/\s+/);
            biddingLines = splitParts.slice(1).map(b => b.trim());
        } else {
            probabilities = parts[1]?.split(/\s+/) || [];
        }

        // Format probabilities
        const formattedProbabilities = probabilities.map(p => parseFloat(p) === -1 ? "-" : p);

        return `
            <tr>
                <td class="deal-cell" data-deal="${deal}">
                    <span class="tooltip">${formatDealForTooltip(deal)}</span>
                    ${deal} 
                    ${biddingLines.length > 0 ? `<br><small>${biddingLines.join("<br>")}</small>` : ""}
                </td>
                ${formattedProbabilities.slice(0, probabilityCount).map(p => `<td class="right-align">${p}</td>`).join('')}
            </tr>
        `;
    }).join('');

    // Construct final HTML
    return `
        <h3 class="samples" onclick="toggleSamples('${tableId}')">
            <strong>Samples(${samples.length}):</strong>
        </h3>
        <div id="${tableId}" class="hidden">
            <table border="1">
                <thead><tr>${headers}</tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>
    `;
}

// Function to format the deal into the 4 hands
function formatDealForTooltip(deal) {
    const hands = deal.split(" "); // Assuming hands are separated by "|"
    if (hands.length !== 4) return "Invalid deal format";

    return `
        <div class="tooltip-content">
            <strong>N:</strong> ${hands[0]}<br>
            <strong>E:</strong> ${hands[1]}<br>
            <strong>S:</strong> ${hands[2]}<br>
            <strong>W:</strong> ${hands[3]}
        </div>
    `;
}
