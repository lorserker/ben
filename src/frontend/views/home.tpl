<!DOCTYPE html>
<html lang="en">

	<head> 
    <meta charset="utf-8">
		<title>BEN - The open source bridge engine</title> 
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <link rel="stylesheet" href="/app/style.css">
    <script>
        function copyToClipboard(idx) {
        const bbaText = document.getElementById('bbaText'+idx);
        const text = bbaText.textContent.trim();

        const el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);

        // You can add a message or any other functionality after copying
        alert('Copied to clipboard: ' + text);
        }
    
        function validateForm(name) {
            var textareaValue = document.getElementById(name).value.trim();
            if (textareaValue === "") {
                alert("Please enter some input before submitting.");
                event.preventDefault()
                return false; // Prevent form submission
            }
            return true; // Allow form submission
        }
        </script>
	</head> 
<body> 
<div class="center">
    <h1>Play with BEN</h1>
</div>

<div class="container">
  <h2>Play this deal: </h2>

  <div class="content">
    <div class="border">
    <form id="form1">
        <label for="board">Board:</label>
        <input type="text" name="board"><br>
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
            <option value="N-S">NS</option>
            <option value="E-W">EW</option>
            <option value="Both">Both</option>
        </select><br>
        <label for="deal">Deal in this format: J2.QJ53.T8754.84 64.K62.Q62.AJT65 KQ985.A.A93.K732 AT73.T9874.KJ.Q9:</label>
        <textarea id="deal" name="dealtext" cols="40"></textarea><br>
        <button type="submit" class="submit-button" data-form="form1" onclick="return validateForm('deal')">Play from text</button>    
    </form>
    </div>
    <br>
    <div class="border">
    <form id="form2">
        <label for="deal">PBN:</label>
        <textarea id="dealpbn" name="dealpbn" cols="40"  rows="6"></textarea><br>
        <button type="submit" class="submit-button" data-form="form2" onclick="return validateForm('dealpbn')">Play from PBN</button>    
    </form>
    </div>
    <br>
    <div class="border">
    <form id="form6">
        <label for="deal">BSOL:</label>
        <textarea id="dealbsol" name="dealbsol" cols="40"  rows="3"></textarea><br>
        <button type="submit" class="submit-button" data-form="form6" onclick="return validateForm('dealbsol')">Play from BSOL</button>    
    </form>
    </div>
    <br>
    <div class="border">
    <form id="form3">
        <label for="deal">LIN:</label>
        <textarea id="deallin" name="deallin" cols="40" rows="3"></textarea><br>
        <button type="submit" class="submit-button" data-form="form3" onclick="return validateForm('deallin')">Play from LIN</button>    
    </form>
    </div>
    <br>
    <div class="border">
    <form id="form4">
        <label for="deal">BBA:</label>
        <textarea id="dealbba" name="dealbba" cols="40"></textarea><br>
        <button type="submit" class="submit-button" data-form="form4" onclick="return validateForm('dealbba')">Play from BBA</button>    
    </form>
    </div>
    <br>
    <div class="border">
    <form id="form5">
        <label for="board">Board:</label>
        <input type="text" name="board"><br>
        <button type="submit" class="submit-button" data-form="form5">Play random</button>    
    </form>
    </div>
</div>
</div>

<div class="container">
  <h2>Seatings</h2>

  <div class="content">
    <div class="inner-div">
    <label for="name">Name:</label>
    <input type="text" id="name" data-default=""><br>
    Controlled by human: <br>
    <input type="checkbox" id="N" data-default="false"><label for="N">North</label>
    <input type="checkbox" id="E" data-default="false"><label for="E">East</label>
    <input type="checkbox" id="S" data-default="true"><label for="S">South</label>
    <input type="checkbox" id="W" data-default="false"><label for="W">West</label><br>
    </div>
    <div class="inner-div">
    Other options: <br>
    <input type="checkbox" id="H" data-default="true"><label for="H">Human declares</label><br>
    <input type="checkbox" id="V" data-default="false"><label for="H">Show all human hands</label><br>
    <input type="checkbox" id="R" data-default="false"><label for="H">Rotate deal</label><br>
    </div>
    <div class="inner-div">
    Automation<br>
    <input type="checkbox" id="A" data-default="true"><label for="T">Autocomplete trick after
    <select id="T">
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2" selected>2</option>
        <option value="5">5</option>
        <option value="10">10</option>
    </select>
    seconds</label><br>
    <input type="checkbox" id="C" data-default="false"><label for="C">Continue to next deal (Requires board number)</label><br>
  </div>
</div>
</div>
<div class="container">
  <h2>Previous played deals</h2>

  <div class="content">
<ul>
% for index, deal in enumerate(deals):
    <li title="{{deal['feedback']}}" class="quality-{{deal['quality']}}">
        <span>{{deal['board_no_index']}}  <a href="/app/viz.html?deal={{deal['deal_id']}}{{deal['board_no_ref']}}">{{deal['contract']}}{{deal.get('trick_winners_count', '')}}</a></span>&nbsp;&nbsp;
        <span><a href="{{deal['delete_url']}}">delete</a></span><br>
        <span class="bba" title="Copy this">BBA=<span id="bbaText{{index}}">{{deal['bba']}}&nbsp;<i class="fas fa-copy" onclick="copyToClipboard({{index}})"></i>
        </span>
        </span>
    </li>
% end
</ul>
</div>
</div>

<script>

    // Retrieve the input field element
    const inputField = document.getElementById('name');

    // Check if there's a value in localStorage, if so, set the input field value to that
    if (localStorage.getItem('inputValue')) {
        inputField.value = localStorage.getItem('inputValue');
    }

    // Add an event listener to store the input field value in localStorage when it changes
    inputField.addEventListener('input', function() {
        localStorage.setItem('inputValue', inputField.value);
    });

    document.addEventListener("DOMContentLoaded", function() {
        var dealTextarea = document.getElementById("deal");
        dealTextarea.addEventListener("input", function() {
            this.value = this.value.toUpperCase();
        });
    });

    // Retrieve the dropdown element
    const dropdown = document.getElementById('T');

    // Check if there's a value in localStorage, if so, set the dropdown value to that
    if (localStorage.getItem('selectedValue')) {
      dropdown.value = localStorage.getItem('selectedValue');
    }

    // Add an event listener to store the selected value in localStorage when the dropdown changes
    dropdown.addEventListener('change', function() {
      localStorage.setItem('selectedValue', dropdown.value);
    });

// Get reference to the checkboxes and forms
const checkbox1 = document.getElementById('N');
const checkbox2 = document.getElementById('E');
const checkbox3 = document.getElementById('S');
const checkbox4 = document.getElementById('W');
const checkbox5 = document.getElementById('H');
const checkbox6 = document.getElementById('A');
const checkbox7 = document.getElementById('C');
const checkbox8 = document.getElementById('R');
const checkbox9 = document.getElementById('V');

// Function to save checkbox state in localStorage
function saveCheckboxState(checkboxId, checked) {
    localStorage.setItem(checkboxId, checked);
}

// Function to load checkbox state from localStorage or set default
function loadCheckboxState(checkboxId, defaultChecked) {
    const savedCheckboxState = localStorage.getItem(checkboxId);

    // If no value exists in localStorage, set a default value
    if (savedCheckboxState === null) {
        checkboxId.checked = defaultChecked
        saveCheckboxState(checkboxId, defaultChecked); // Set to default value
        return defaultChecked;
    } else {
        return savedCheckboxState === 'true';
    }
}

// Load checkbox states when the page is loaded
document.addEventListener('DOMContentLoaded', function() {
    const checkboxes = document.querySelectorAll('input[type="checkbox"]');

    checkboxes.forEach(checkbox => {
        const defaultChecked = checkbox.getAttribute('data-default') === 'true';
        checkbox.checked = loadCheckboxState(checkbox.id, defaultChecked);

        checkbox.addEventListener('click', function() {
            saveCheckboxState(checkbox.id, this.checked);
        });
    });
});

const submitButtons = document.querySelectorAll('.submit-button');

// Variable to store the selected form
let selectedForm = null;


// Function to include checkbox values in form submission
function includeCheckboxValues(event) {
    event.preventDefault(); // Prevents the default form submission

    if (selectedForm) {
        const formData = new FormData(selectedForm);
        const checkboxes = [checkbox1, checkbox2, checkbox3, checkbox4, checkbox5, checkbox6, checkbox7, checkbox8, checkbox9];
        checkboxes.forEach(checkbox => {
            if (checkbox.checked) {
                formData.append(checkbox.id, checkbox.value);
            }
        });

        // Append input field value
        const inputField = document.getElementById('name');
        formData.append('name', inputField.value);

        // Append dropdown selected value
        const dropdown = document.getElementById('T');
        formData.append('T', dropdown.value);

        // You can submit the form data using fetch or XMLHttpRequest here
        // For example:
        fetch('/submit', {
             method: 'POST',
             body: formData
        })
        .then(response => {
            if (response.redirected) {
                // If the response indicates a redirect, you can handle it here
                // For example, you can redirect the user to the new location
                window.location.href = response.url;
            }        
        })
        .catch(error => {
            // Handle error
        });
        console.log('Form Data:', formData); // For demonstration purposes
    };
}

// Function to update selectedForm variable based on button click
function updateSelectedForm(event) {
    selectedForm = document.getElementById(this.dataset.form);
}

// Attach the includeCheckboxValues function to form submit events
submitButtons.forEach(button => {
    button.addEventListener('click', updateSelectedForm);
    button.addEventListener('click', includeCheckboxValues);
});

</script>
</body> 
</html> 
