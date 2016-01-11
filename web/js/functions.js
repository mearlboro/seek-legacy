$(document).ready(function(){
    $('#textbox').on('keyup', function(event) {
        if (event.keyCode == 13) {
            $("#answer").css('display', 'block');
        }
    });
});
