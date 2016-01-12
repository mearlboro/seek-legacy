$(document).ready(function(){
    $('.main-input').on('keyup', function(event) {
        if (event.keyCode == 13) {
            $("#result").css('display', 'block');
            $("#question").append("<a href=\'#result\'></a>");
        }
    });
    $('#click4info').click(function() {
      $('#more').css('display', 'block');

    });
});
