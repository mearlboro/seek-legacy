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


/* toggle for mic/keyboard button */
function toggleInput() {
    but = $('#input-toggle').get(0);
    classList = but.className.split(' ');

    if (classList[3] == 'fa-microphone') {
        but.className = 'btn btn-lg fa fa-keyboard-o';
        if (annyang) {
            annyang.start();
        }
    }
    if (classList[3] == 'fa-keyboard-o') {
        but.className = 'btn btn-lg fa fa-microphone';
        if (annyang) {
            annyang.abort();
        }
    }
}

