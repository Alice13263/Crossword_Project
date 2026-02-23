//File to activate dark mode when button is clicked
let overlay;
let accessibility_button;
//Checks the mode once the page has loaded
window.addEventListener("DOMContentLoaded", ()=>{
    overlay = document.getElementById("accessibility_menu_overlay");
    accessibility_button = document.getElementById("accessibility_button");
    const dark_mode_button = document.getElementById("colour_mode");
    //Triggers the overlay if the menu button is clicked
    accessibility_button.onclick = function(){
        if (overlay.style.display == "block"){
            overlay.style.display = "none";
        }
        else{
            overlay.style.display = "block";
        }
    };
    //Sets dark mode if in light mode, and vice versa
    if (dark_mode_button){
        dark_mode_button.addEventListener("change", function(){
            if (dark_mode_button.checked){
                setMode("dark_mode");
            }
            else{
                setMode("light_mode");
            }
        });
    }
    displaySettings();
});
//Function to remove class names, and assign the correct mode
function setMode(mode){
    document.body.classList.remove("light_mode", "dark_mode");
    document.body.classList.add(mode);
    localStorage.setItem("mode", mode);
};
//Function that retrieves the current mode, sets the default if needed, and changes the page on click
function displaySettings(){
    let mode = localStorage.getItem("mode");
    if (!mode){
        mode = "light_mode"
    }
    document.body.classList.add(mode);
    const dark_mode_button = document.getElementById("colour_mode");
    if (dark_mode_button){
        dark_mode_button.checked = (mode == "dark_mode")
    }
};