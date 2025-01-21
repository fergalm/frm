from pathlib import Path 
import pandas as pd 
import datetime 
import os 

def writeFancyHtml(df, title, outfile):
    """Write dataframe to a fancy searchable  html table and write to a  file
    
    Inputs
    ------------
    df
        (pd.DataFrame) Dataframe to write to html 
    title
        (str) Title page of html page 
    outfile
        (str) Name of file to write. 
        
    
    Returns 
    -------------
    **None** 
    
    
    Note: This function writes the table to a dedicated html page. With a bit of work, you
    could figure out how to embaed the table is a larger page
    """
    html = df.to_html(table_id = "myTable", index=False, escape=False, classes=["stripe", "hover", "cell-border"])

    cwd = Path(__file__).parent 
    templateFile = os.path.join(cwd, "template.html")
    with open(templateFile) as fp:
        template = fp.read()


    #Wrap dataframe with required boilerplate
    now = datetime.datetime.now()
    html = template % ("Notable Votes", now, html)
    with open(outfile, 'w') as fp:
        fp.write(html)


def makeListOfLinks(url: pd.Series, names: pd.Series):
    """Make a list of strings matching "<A href="url", target=_blank">name</A>

        Turns two series of a dataframe into a list of links to put in a html table
    
        Inputs
        -----------
        url 
            (pd.Series of strings) urls to include 
        names 
            (pd.Series of strings) link text to include 
    
        Returns 
        ----------
        List of strings
    
    """
    return lmap(lambda x, y: f"<A href='{x}' target='_blank'>{y}</A>", url, df.BillNumber)

