/* grab the element and add event listener for when animations end */
eye = $('#eye').get(0)

eye.addEventListener('webkitAnimationEnd', function(){
    mood = this.style.animationName;
    console.log("end of animation" + mood);
    this.style.webkitAnimationName = '';
    this.style.mozAnimationName    = '';
    this.style.animationName       = '';
    console.log(mood)
    if(mood != '') {
        this.className = mood;
    }
}, false);

function changeMood(mood) {
    eye.style.webkitAnimationName = mood; 
    eye.style.mozAnimationName    = mood; 
    eye.style.animationName       = mood; 
}

/* TODO: replace these with calls from the server */
document.getElementById('neutral').onclick  = changeMood('neutral');
document.getElementById('happy').onclick    = changeMood('happy');
document.getElementById('sad').onclick      = changeMood('sad');
document.getElementById('thinking').onclick = changeMood('thinking');



