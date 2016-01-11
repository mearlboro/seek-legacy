$(document).ready(function(){
    $('.main-input').on('keyup', function(event) {
        if (event.keyCode == 13) {
            $("#answer").css('display', 'block');
        }
    });
});
