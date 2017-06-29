#! /usr/bin/python
''' Module for interacting with IPython notebooks '''

def run_nb(workdir, fileroot, notebook):

    from shutil import copy
    from rtpipe import get_notebook
    from subprocess import call

    os.environ['fileroot'] = fileroot
    if agdir: #git directory
        os.environ['agdir'] = agdir
    
    os.chdir(workdir) 
    copy(nb, '{0}/{1}.ipynb'.format(workdir, fileroot))

    cmd = 'jupyter nbconvert {0}.ipynb --inplace --execute --to notebook --allow-errors --ExecutePreprocessor.timeout=3600'.format(fileroot).split(' ')
    status = call(cmd)
    
    cmd = 'jupyter trust {0}.ipynb'.format(fileroot).split(' ')
    status = call(cmd)
    
    return True
    
    
