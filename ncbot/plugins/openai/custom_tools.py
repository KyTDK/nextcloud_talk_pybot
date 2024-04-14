
from langchain.tools import tool
import nc_py_api
import ncbot.config as ncconfig

nc = nc_py_api.Nextcloud(nextcloud_url=ncconfig.cf.base_url,
                         nc_auth_user=ncconfig.cf.username, nc_auth_pass=ncconfig.cf.password)

@tool
def WriteFile(file_name:str, content:str):
    """Create and save information into a text file"""
    nc.files.upload(file_name, content)