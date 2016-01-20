$(document).ready(function() {
    $('.main-input').on('keyup', function(event) {
        if (event.keyCode == 13) {
            $("#result").css('display', 'block');
            $("#query").append("<a href=\'#result\'></a>");
        }
    });

    $('#click4info').click(function() {
        $('#more').css('display', 'block');
    });

    $('#upload').click(function(event) {
        $('#littlebox').lightbox_me({
            centered: true,
            onLoad: function() {
            }
        });
        event.preventDefault();
    });
});
