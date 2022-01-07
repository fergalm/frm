
"""Sync cwd of a project repo to AWS"""

import subprocess
import argparse
import smtplib
from email.mime.text import MIMEText

def parse_input_arguments():
    parser = argparse.ArgumentParser(description="Sync a project to AWS")

    parser.add_argument("direction", help="{push|pull}")
    parser.add_argument("-d", "--delete", help="Same as AWS --delete option", action="store_true")
    parser.add_argument("-r", "--dryrun", help="Same as AWS --dry-run option", action="store_true")
    return parser


def sync(direction, project, path,
         bucket="orbital-projects",
         delete=False,
         dry_run=False):

    if direction not in ["push", "pull"]:
        raise ValueError("Illegal value for direction. Must be {push|pull}")

    args = "--exclude=*.TIF --exclude cache* --exclude=.cache* --exclude=*.tgz --exclude=*.zip"
    if direction == "push":
        cmd = "aws s3 sync %s s3://%s/%s %s" %(path, bucket, project, args)
    else:
        cmd = "aws s3 sync s3://%s/%s %s %s" %(bucket, project, path, args)

    if delete:
        cmd = "%s --delete" %(cmd)

    #Issue command
    if not dry_run:
        msg = system(cmd)


    #Issue command with dryrun, capture output.
    #If actual command is successful, this will return nothing
    cmd = "%s --dryrun" %(cmd)
    msg = system(cmd)

    return msg


def email_on_error(msg, toaddress):
    raise NotImplementedError("Doesn't work yet")

    if len(msg) > 0:
        print "Error syncing directory"
        print msg

    #sender = "fergal.cron@gmail.com"
    sender = "fergal@orbitalinsight.com"
    #sender = toaddress
    body = MIMEText(msg)
    body['To'] = toaddress
    body['From'] = sender
    body['Subject'] = "Error syncing directory"

    s = smtplib.SMTP_SSL('smtp.gmail.com', 465)
    s.ehlo()
    s.sendmail(sender, [toaddress], body.as_string())
    s.quit()


def system(command):
    try:
        output = subprocess.check_output(command.split(),\
                    stderr=subprocess.STDOUT)
        return output
    except subprocess.CalledProcessError as e:
        return str(e)




