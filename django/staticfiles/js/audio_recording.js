// Phrase to say before Seek recognises audio
if (annyang) {    

    var commands = {
        '(seek) (sick) *tag' : function(tag){
            document.getElementById("id_query").value = tag.concat("?");
            document.getElementById("id_query_form").submit();
        },
    };


//  annyang.debug();
    annyang.setLanguage('en');

    annyang.addCommands(commands);
//    annyang.start();
}


