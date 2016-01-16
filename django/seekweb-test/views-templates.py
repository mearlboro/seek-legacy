from django.shortcuts import render, HttpResponse
from .forms import DocumentForm, QueryForm
from .models import Document
from django.template import RequestContext, loader
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from .filehandler import handle_file, handle_question, handle_command

# {% mood_var = {'neutral': load_neutral, 'happy': load_happy, 'sad': load_sad, 'thinking': load_thinking} %} -->

def index(request):

	# t = loader.get_template('seek/commands.html')
	mood_var = 'neutral'	
	return render(request, 'seek/index.html', {'mood_var':mood_var}, context_instance=RequestContext(request))


def commands(request):
	if 'files' not in request.session:
		request.session['files'] = []
	if 'name' not in request.session:
		request.session['name'] = 'Stranger'
	if answer == ' ':
		answer = "Hello, " + request.session['name']
	answer_display = "display: 'none'"
	confirmation = ""

	if request.method == 'GET':
		queryform = QueryForm()
		docform = DocumentForm()
		return render(request, 'seek/commands.html', {'docform': docform, 'queryform':queryform, 'display_answer':answer_display, 'answer':answer, 'confirmation':confirmation})

	if request.method == 'POST':
		docform = DocumentForm(request.POST, request.FILES)
		queryform = QueryForm(request.POST)
		text = ""

		if docform.is_valid():
			f = request.FILES['docfile']
			(retfile, confirmation) = handle_file(f, f.name)
			request.session["files"] += [retfile] 
		if queryform.is_valid():
			query = queryform.cleaned_data['query']
			query_response = handle_command(query, request.session['files'])
			answer = query_response
			answer_display = "display: 'block'"	

	queryform = QueryForm()
	docform = DocumentForm()

	return render(request, 'seek/index.html', {'docform': docform, 'queryform':queryform, 'confirmation':confirmation, 'answer':answer, 'answer_display':answer_display, 'more':'', 'mood_var': mood_var}, context_instance=RequestContext(request))
	


def answer(request):
	if 'files' not in request.session:
		request.session['files'] = []
	if 'name' not in request.session:
		request.session['name'] = 'Stranger'
	if answer == ' ':
		answer = "Hello, " + request.session['name']
	answer_display = "display: 'none'"
	more = ''

	return render(request, 'seek/answer.html', {'answer':answer, 'answer_display':answer_display, 'more':more})
		

