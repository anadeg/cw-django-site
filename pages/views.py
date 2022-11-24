import mysql.connector
from django.shortcuts import render

import os
from pathlib import Path

import json



def home_page_view(request, *args, **kwargs):
    path = os.path.join('home_page.html')
    return render(request, path, {})


def query(q):
    BASE_DIR = Path(__file__).resolve().parent.parent

    with open(os.path.join(BASE_DIR, 'cw_site', 'security', 'security.json'), 'r') as f:
        data = json.load(f)

    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password=data["password"],
        database='COURSEWORK'
    )

    cursor = mydb.cursor()
    cursor.execute(q)
    result = cursor.fetchall()
    return result


def search_page(request, *args, **kwargs):
    concept = request.GET.get('concept_search')
    path = os.path.join('search.html')
    select = f"""SELECT concept, description FROM concepts
                 WHERE concept LIKE '%{concept}%';"""
    try:
        c = query(select)
        concepts = {c[i][0]: c[i][1] for i in range(len(c))}
        context = {
            "concepts": concepts
        }
        return render(request, path, context)
    except Exception:
        return render(request, path, {})


if __name__ == '__main__':
    pass

