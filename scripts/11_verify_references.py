"""Verify reference metadata and DOIs against Crossref (run 30 Sept 2026).
Output: refs/reference_check.csv"""
import json, time, urllib.parse, urllib.request
import pandas as pd

Q = [
 ("Li Sinha 2025", "Gauging inflation aversion through the lens of consumer sentiment", "Economics Letters"),
 ("Kahneman Tversky 1979", "Prospect theory: an analysis of decision under risk", "Econometrica"),
 ("Tversky Kahneman 1991", "Loss aversion in riskless choice: a reference-dependent model", "Quarterly Journal of Economics"),
 ("Brachinger 2008", "A new index of perceived inflation: assumptions, method, and application to Germany", "Journal of Economic Psychology"),
 ("Drager Menz Fritsche 2014", "Perceived inflation under loss aversion", "Applied Economics"),
 ("Ahn Tsuchiya 2022", "Consumer's perceived and expected inflation in Japan: irrationality or asymmetric loss", "Empirical Economics"),
 ("DAcunto et al 2021", "Exposure to grocery prices and inflation expectations", "Journal of Political Economy"),
 ("Souleles 2004", "Expectations, heterogeneous forecast errors, and consumption: micro evidence from the Michigan consumer sentiment surveys", "Journal of Money, Credit and Banking"),
 ("Bachmann Berg Sims 2015", "Inflation expectations and readiness to spend: cross-sectional evidence", "American Economic Journal: Economic Policy"),
 ("Carroll Fuhrer Wilcox 1994", "Does consumer sentiment forecast household spending? If so, why?", "American Economic Review"),
 ("Coibion Gorodnichenko Weber 2022", "The expected, perceived, and realized inflation of U.S. households before and during the COVID19 pandemic", "IMF Economic Review"),
 ("Genesove Mayer 2001", "Loss aversion and seller behavior: evidence from the housing market", "Quarterly Journal of Economics"),
 ("Barberis 2013", "Thirty years of prospect theory in economics: a review and assessment", "Journal of Economic Perspectives"),
 ("Curtin 2019", "Consumer Expectations: Micro Foundations and Macro Impact", ""),
 ("Coibion Gorodnichenko Weber JEP", "The subjective inflation expectations of households and firms", "Journal of Economic Perspectives"),
 ("Abdellaoui Bleichrodt Paraschiv 2007", "Loss aversion under prospect theory: a parameter-free measurement", "Management Science"),
 ("Kahneman Knetsch Thaler 1990", "Experimental tests of the endowment effect and the Coase theorem", "Journal of Political Economy"),
]
rows = []
for key, title, journal in Q:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query.bibliographic": title + " " + journal, "rows": 3,
         "select": "DOI,title,volume,issue,page,article-number,issued,container-title,author,type"})
    it = json.load(urllib.request.urlopen(url, timeout=30))["message"]["items"]
    for x in it:
        if x.get("type") in ("journal-article", "book", "monograph") and x.get("title"):
            break
    rows.append({"key": key, "doi": x.get("DOI"), "title": x.get("title", [""])[0],
                 "journal": (x.get("container-title") or [""])[0], "year": x["issued"]["date-parts"][0][0],
                 "volume": x.get("volume"), "issue": x.get("issue"), "pages": x.get("page") or x.get("article-number"),
                 "authors": "; ".join(f"{a.get('family')}, {a.get('given')}" for a in x.get("author", []))})
    time.sleep(0.5)
df = pd.DataFrame(rows)
df.to_csv("refs/reference_check.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 70)
print(df.to_string())
