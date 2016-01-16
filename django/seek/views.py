from django.shortcuts import render, HttpResponse
from .forms import DocumentForm, QueryForm
from .models import Document
from django.template import RequestContext, loader
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from .filehandler import handle_file, handle_question, handle_user_input#, handle_command
from .commandhandler import handle_command
import logging

logger = logging.getLogger('handler')

def index(request):
        if 'files' not in request.session:
                request.session['files'] = []
        request.session['name'] = 'Stranger'
     
        if 'name' not in request.session:
                request.session['name'] = 'Stranger'
        if 'history' not in request.session:
                request.session['history'] = ''

        answer = "Hello, " + request.session['name']
        answer_display = "none"
        confirmation = ''
        mood_var = 'neutral'    
        more = ''
        if request.method == 'POST':
                docform = DocumentForm(request.POST, request.FILES)
                queryform = QueryForm(request.POST)
                text = ""

                if docform.is_valid():
                        f = request.FILES['docfile']
                        (retfile, confirmation) = handle_file(f, f.name)
                        request.session["files"] += [retfile]
                        mood_var = 'thinking'
                if queryform.is_valid():
                        query = queryform.cleaned_data['query']
                        place_holder = query
                        #query_response = handle_question(query, request.session['files'])
                        query_response = handle_user_input(query, request.session['files'], request.session['name'])
                        answer = query_response[1]
                        answer_display = "block"        
                        request.session["name"] = query_response[2]
        queryform = QueryForm()
        docform = DocumentForm()

        return render(request, 'seek/index.html', {'docform': docform, 'queryform':queryform, 'confirmation':confirmation, 'answer':answer, 'answer_display':answer_display, 'more':'', 'mood_var': mood_var}, context_instance=RequestContext(request))
