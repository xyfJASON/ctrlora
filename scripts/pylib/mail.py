import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header


def send_email(
    smtp_server,
    port,
    sender_email,
    sender_password,
    receiver_email,
    subject,
    body,
    body_type="plain",
    attachment_paths=[],
    get_socket_func=None,
):
    """
    Function to send an email with or without attachments, with body content either in plain text or HTML.

    Parameters:
    - smtp_server: Address of the SMTP server.
    - port: Port number for the SMTP server.
    - sender_email: Email address of the sender.
    - sender_password: Password for the sender's email account.
    - receiver_email: Email address of the recipient.
    - subject: Subject of the email.
    - body: Body content of the email.
    - body_type: Type of the body content ('plain' or 'html', default is 'plain').
    - attachment_paths: List of file paths for the attachments (optional).
    """

    if get_socket_func is not None:
        ori_get_socket_func = smtplib.SMTP._get_socket
        smtplib.SMTP._get_socket = get_socket_func

    try:
        try:
            server = smtplib.SMTP(smtp_server, port)
            server.ehlo()
            server.starttls()
            server.ehlo()
        except:
            if 'server' in locals():
                server.quit()
            server = smtplib.SMTP_SSL(smtp_server, port)

        message = MIMEMultipart()
        message['From'] = Header(sender_email, 'utf-8')
        message['To'] = Header(receiver_email, 'utf-8')
        message['Subject'] = Header(subject, 'utf-8')
        message.attach(MIMEText(body, body_type, 'utf-8'))
        for attachment_path in attachment_paths:
            with open(attachment_path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            filename = os.path.basename(attachment_path)
            part.add_header("Content-Disposition", f"attachment; filename={filename}")
            message.attach(part)

        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
    except:
        raise
    finally:
        if get_socket_func is not None:
            smtplib.SMTP._get_socket = ori_get_socket_func
        if 'server' in locals():
            server.quit()
