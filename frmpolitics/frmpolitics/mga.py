from io import StringIO 
import pandas as pd 
import requests 

class MgaClient:
    """Interface for querying known parts of the API of the Maryland General Assembly"""
    
    def __init__(self):
        self.url = "https://mgaleg.maryland.gov"
    
    def get_summary_file(self, year) -> pd.DataFrame:
        url = [
            self.url, 
            f"{year}rs/misc/billsmasterlist",
            "legislation.json"
        ]
        url = "/".join(url)
        # print(url)
        # return 
        response = query(url, None)
        json = response.json()
        df = pd.DataFrame(json)
        return df 

    def get_original_bill_text_url(self, year, billNumber):
        url = [ self.url,
               "mgawebsite/Legislation/Details/",
               f"{billNumber}?ys={year}RS",
        ]
        return "/".join(url)
    
    def get_vote_tally_url(self, year, chamber, votenum):
        assert chamber in ["house", "senate"]
        url = [
            self.url,
            f"{year}",
            f"votes/{chamber}/{votenum:04d}.pdf"
        ]
        return "/".join(url)
        
    
def query(url, outpath=None):
    """Query the url and write the result to a file 
    
    Inputs
    ----------
    url 
        URL to query 
    outpath
        Full path to save result to. 
        
    Returns 
    -----------
    (int) status code.
    
    Note that if outpath is not set, a requests response object is returned
    that can be manipulated however you want. If a path is specified,
    the contents of the response are written to that path in binary mode.
    """
    
    response = requests.get(url) 
    response.raise_for_status()
    
    if outpath is None:
        return response 

    #Create directory if necessary 
    outdir = os.path.join(os.path.split(outpath)[:-1])
    os.makedirs(outdir, exist_ok=True)
        
    with open(outfile, 'wb') as fp:
        fp.write(response.content)
    return response.status_code
    


def convert_pdf_to_html(pdfpath):
    """Convert a pdf file to a html file"""
    outpath = re.subn("pdf", "html", pdfpath)[0]
    
    os.makedirs(outpath, exist_ok=True)
    cmd = f"pdftohtml {pdfpath} {outpath} > /dev/null"
    os.system(cmd)


