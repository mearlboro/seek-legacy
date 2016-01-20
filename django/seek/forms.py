from django import forms

class QueryForm(forms.Form):
    query = forms.CharField(widget=forms.TextInput(attrs={'placeholder': ''}), label = '')

class DocumentForm(forms.Form):
    docfile = forms.FileField(label = "")
