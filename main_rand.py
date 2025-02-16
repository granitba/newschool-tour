import gzip
import json
import math
import multiprocessing as mp
import os
import random
import re
import socket
import string
import time
import urllib.parse
from datetime import datetime  # Importojmë datetime për caktimin e kohës
from multiprocessing import Pool
from captcha import main

from bs4 import BeautifulSoup
from curl_cffi import requests

from captcha.main import predict_image, load_model

# Load random config_mine.json
random_config = random.choice(os.listdir('configs'))
def load_config():
    with open('configs/' + random_config, 'r') as file:
        print(f"Using config {random_config}")
        return json.load(file)


config = load_config()  # Ngarkon konfigurimin nga config_mine.json

# Disable SSL warnings (for development purposes only, not recommended for production)
_base_chars_for_rand_api_key = list(string.hexdigits.lower())
real_key = "0ccf26489d12118c"
rand_key = "".join(random.choices(_base_chars_for_rand_api_key, k=len(real_key)))


proxies = {
    # 'http': 'http://127.0.0.1:8080',
    #  'https': 'http://127.0.0.1:8080',
}
headers = {
    "Cache-Control": "max-age=0",
    "Sec-Ch-Ua": "\"Chromium\";v=\"95\", \";Not A Brand\";v=\"99\"",
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": "\"Windows\"",
    "Upgrade-Insecure-Requests": "1",
    "Origin": "https://service2.diplo.de",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-User": "?1",
    "Sec-Fetch-Dest": "document",
    "Referer": "https://service2.diplo.de/rktermin/extern/appointment_showMonth.do",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close"
}
BATCH_SIZE = 50         # Number of logs per batch
FLUSH_INTERVAL = 6     # Time in seconds to send logs even if the batch is not full
LOG_SERVER_URL = f"http://{config['log_server_ip']}:{config['log_server_port']}/gelf"
LOCAL_IP = socket.gethostbyname(socket.gethostname())
MAX_RETRIES = 15
MAX_CAPTCHA_RETRIES = 15  # Retry limit for missing CAPTCHA
BACKOFF_FACTOR = 1
STATUS_FORCELIST = [500, 502, 503, 504]



def log_sender(queue):
    """Process that sends logs from the queue to the remote server in batches."""
    os.nice(10)
    batch = []
    last_flush_time = time.time()

    while True:
        try:
            log_entry = queue.get(timeout=FLUSH_INTERVAL)
            if log_entry is None:  # Sentinel value to stop the process
                if batch:
                    send_batch(batch)  # Send remaining logs before exiting
                break

            batch.append(log_entry)

            # Send the batch if the size limit is reached
            if len(batch) >= BATCH_SIZE:
                send_batch(batch)
                batch = []  # Reset the batch


        except Exception as e:
            # Timeout happens when no new logs arrive, flush the batch if necessary
            if batch and (time.time() - last_flush_time >= FLUSH_INTERVAL):
                send_batch(batch)
                batch = []  # Reset the batch

def send_batch(batch):
    """Compresses and sends a batch of logs."""
    try:
        newline_delimited_logs = "\n".join(json.dumps(logs,ensure_ascii=False) for logs in batch).encode("utf-8")
        compressed_logs = gzip.compress(newline_delimited_logs)
        log_headers = {"Content-Encoding": "gzip", "Content-Type": "application/json"}
        response = requests.post(LOG_SERVER_URL, headers=log_headers, data=compressed_logs)
        response.raise_for_status()
        print(f"Sent {len(batch)} logs successfully.")
    except Exception as e:
        print(f"Failed to send batch: {e}")

def retry_request(session, method, url, queue, max_retries=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, **kwargs):
    """A retry wrapper for HTTP requests that always retries, regardless of the error."""
    attempt = 1
    response = None  # Initialize response to None to avoid referencing before assignment

    while attempt <= max_retries:
        try:
            # Make the request (GET or POST)
            if method == "GET":
                response = session.get(url, **kwargs)
            elif method == "POST":
                response = session.post(url, **kwargs)

            # Check if the response was successful (2xx status)
            response.raise_for_status()

            # Log success
            queue.put({
                "level": "INFO",
                "message": f"Request to {url} successful on attempt {attempt}",
                "timestamp": time.time(),
            })

            return response  # Return the response if successful

        except requests.exceptions.RequestException as e:
            # Log the error, regardless of the type
            error_message = f"Error on attempt {attempt} for {url}: {e}. Retrying..."

            queue.put({
                "level": "ERROR",
                "message": error_message,
                "timestamp": time.time(),
            })

        # Wait before retrying (exponential backoff)
        time.sleep(backoff_factor * (attempt / 2))
        attempt += 1

    # If all retries are exhausted, return the last response (or None)
    queue.put({
        "level": "ERROR",
        "message": f"Failed to complete request after {max_retries} attempts.",
        "timestamp": time.time(),
    })

    return response

def wait_process(target_time_str):
    target_time = datetime.strptime(target_time_str, "%H:%M:%S.%f").time()
    while datetime.now().time() < target_time:
        print(f"Processing waiting for {target_time}. Current time: {datetime.now().time()}")
        time.sleep(0.01)

# Function to wait until a specific time (hour, minute, second)
def wait_until_specific_time(target_time_str, queue):
    target_time = datetime.strptime(target_time_str, "%H:%M:%S").time()
    log_interval = 80
    i = 0

    # load captcha model
    load_model()

    while datetime.now().time() < target_time:
        print(f"Po pret kohen e caktume {target_time}. Koha e tanishme: {datetime.now().time()}")
        if i % log_interval == 0:
            log_entry = {
                "server": LOCAL_IP,
                "level": "INFO",
                "message": f"Po pret kohen e caktume {target_time}. Koha e tanishme: {datetime.now().time()}",
                "timestamp": time.time(),
            }
            # Add log entry to the queue for centralized logging
            queue.put(log_entry)
        i += 1
        time.sleep(0.1)


def solve_captcha(image_base64, session, queue):
    """Solve CAPTCHA using the custom service and log events."""

    encoded_image = urllib.parse.quote(image_base64)
    result = predict_image(image_base64)
    queue.put({
        "level": "INFO",
        "message": f"CAPTCHA successfully solved. Result: {result}",
        "timestamp": time.time()
    })
    print(f"Captcha solved successfully: {result}")
    return result

    # # Use IP and port from config_mine.json
    # url = f"http://{config['captcha_ip']}:{config['captcha_port']}/predict?image={encoded_image}"
    #
    # headers = {'Content-Type': 'application/json'}
    #
    # # Log the CAPTCHA solving attempt
    # queue.put({
    #     "level": "DEBUG",
    #     "message": f"Attempting to solve CAPTCHA using the custom service at {url}.",
    #     "timestamp": time.time()
    # })
    #
    # try:
    #     response = session.post(url, headers=headers, timeout=10, proxies=proxies, verify=False)
    #     response.raise_for_status()
    #
    #     result = response.json()
    #
    #     if 'captchaResult' in result:
    #         captcha_text = result['captchaResult'][0]
    #
    #         # Log success in CAPTCHA solving
    #         queue.put({
    #             "level": "INFO",
    #             "message": f"CAPTCHA successfully solved. Result: {captcha_text}",
    #             "timestamp": time.time()
    #         })
    #
    #         return captcha_text
    #     else:
    #         error_message = "CAPTCHA solving service did not return a valid result."
    #         print(error_message)
    #
    #         # Log missing CAPTCHA result
    #         queue.put({
    #             "level": "INFO",
    #             "message": error_message,
    #             "timestamp": time.time()
    #         })
    #
    #         return None
    #
    # except requests.exceptions.RequestException as e:
    #     error_message = f"Failed to solve CAPTCHA: {e}"
    #     print(error_message)
    #
    #     # Log CAPTCHA service request error
    #     queue.put({
    #         "level": "INFO",
    #         "message": error_message,
    #         "timestamp": time.time()
    #     })
    #
    #     return None



def first_captcha_solver(session, first_captcha_url, date_post_file, queue, post_time=None):
    """Submit the first CAPTCHA and log events to the queue."""
    post_response = None  # Initialize post_response to None to avoid uninitialized reference

    if post_time:
        wait_until_specific_time(post_time, queue)  # Wait until the specified POST time

    # GET request with retry policy and logging
    response = retry_request(
        session, 'GET', first_captcha_url, queue,
        headers=headers, proxies=proxies, verify=False
    )

    if response is None:
        error_message = f"Failed to access the showMonth page after multiple attempts."
        print(error_message)
        queue.put({
            "level": "INFO",
            "message": error_message,
            "timestamp": time.time()
        })
        return None, None

    # Log debug message
    queue.put({
        "level": "DEBUG",
        "message": f"Successfully accessed the showMonth page: {first_captcha_url}",
        "timestamp": time.time()
    })

    # Parse the response and extract CAPTCHA
    soup = BeautifulSoup(response.content, 'html.parser')
    captcha_element = soup.find('div', style=re.compile(r"background:white url\('data:image/jpg;base64,(.+)'\).*"))

    if captcha_element:
        match = re.search(r"background:white url\('data:image/jpg;base64,(.+)'\).*", captcha_element['style'])
        if match:
            base64_image = match.groups(0)[0]

            # Log successful CAPTCHA extraction
            queue.put({
                "level": "DEBUG",
                "message": "CAPTCHA successfully extracted from the showMonth page.",
                "timestamp": time.time()
            })
        else:
            error_message = "CAPTCHA not found in the extracted element."
            print(error_message)

            # Log info message for missing CAPTCHA
            queue.put({
                "level": "INFO",
                "message": error_message,
                "timestamp": time.time()
            })
            return None, None
    else:
        error_message = "CAPTCHA element not found in the showMonth page."
        print(error_message)

        # Log info message for missing CAPTCHA element
        queue.put({
            "level": "INFO",
            "message": error_message,
            "timestamp": time.time()
        })
        return None, None

    # Solve the CAPTCHA
    captcha_text = solve_captcha(base64_image, session, queue)
    if not captcha_text:
        return None, None

    # Simulate a delay before submission
    time.sleep(config['delay_captcha'])

    # Load the form data
    with open(date_post_file, 'r') as json_file:
        data = json.load(json_file)
    data['captchaText'] = captcha_text

    # Log the preparation of the form
    queue.put({
        "level": "DEBUG",
        "message": "Form data prepared with CAPTCHA for submission.",
        "timestamp": time.time()
    })

    first_captcha_post_url = "https://service2.diplo.de/rktermin/extern/appointment_showMonth.do"

    # POST request with retry policy and logging
    post_response = retry_request(
        session, 'POST', first_captcha_post_url, queue,
        data=data, headers=headers, proxies=proxies, verify=False
    )

    if post_response is None:
        error_message = f"Failed to submit the first CAPTCHA form after multiple attempts."
        print(error_message)
        queue.put({
            "level": "INFO",
            "message": error_message,
            "timestamp": time.time()
        })
        return None, None

    # Check if the POST request was successful
    if post_response.status_code == 200:
        success_message = "Successfully submitted the first CAPTCHA form."
        print(success_message)

        # Log success message
        queue.put({
            "level": "INFO",
            "message": success_message,
            "timestamp": time.time()
        })

        return session, post_response

    # Handle non-200 status codes
    error_message = f"Failed to submit the first CAPTCHA form. Status code: {post_response.status_code}"
    print(error_message)

    queue.put({
        "level": "INFO",
        "message": error_message,
        "timestamp": time.time()
    })

    return None, None


def process_show_form(session, show_form_url, passport_json_file, queue, get_time=None):
    """Process the showForm page, extract CAPTCHA, and submit form data, retrying if CAPTCHA is not found."""

    MAX_CAPTCHA_RETRIES = 3  # Retry limit for missing CAPTCHA
    captcha_retry_attempt = 0

    post_response = None  # Initialize post_response to None to avoid uninitialized reference

    if get_time:
        wait_until_specific_time(get_time, queue)  # Wait until the specified GET time

    while captcha_retry_attempt < MAX_CAPTCHA_RETRIES:
        # GET request with retry policy and logging
        response = retry_request(
            session, 'GET', show_form_url, queue,
            headers=headers, proxies=proxies, verify=False
        )

        if response is None:
            error_message = f"Failed to get showForm page after multiple attempts."
            print(error_message)
            queue.put({
                "level": "INFO",
                "message": error_message,
                "timestamp": time.time()
            })
            return None  # Exit early if GET request fails after retries

        # Log successful response retrieval
        queue.put({
            "level": "DEBUG",
            "message": f"Received response from {show_form_url} for passport {passport_json_file}",
            "timestamp": time.time()
        })

        # Parse the page and extract CAPTCHA
        soup = BeautifulSoup(response.content, 'html.parser')
        captcha_element = soup.find('div', style=re.compile(r"background:white url\('data:image/jpg;base64,(.+)'\).*"))

        if captcha_element:
            match = re.search(r"background:white url\('data:image/jpg;base64,(.+)'\).*", captcha_element['style'])
            if match:
                base64_image = match.groups(0)[0]

                # Log successful CAPTCHA extraction
                queue.put({
                    "level": "DEBUG",
                    "message": "Successfully extracted CAPTCHA from the showForm page.",
                    "timestamp": time.time()
                })
                break  # Exit the retry loop once CAPTCHA is found
            else:
                error_message = "CAPTCHA not found in the extracted element."
                print(error_message)
                queue.put({
                    "level": "INFO",
                    "message": error_message,
                    "timestamp": time.time()
                })
        else:
            error_message = f"CAPTCHA element not found on the page for passport {passport_json_file} attempt # {captcha_retry_attempt}."
            print(error_message)
            queue.put({
                "level": "INFO",
                "message": error_message,
                "timestamp": time.time()
            })

        captcha_retry_attempt += 1
        if captcha_retry_attempt < MAX_CAPTCHA_RETRIES:
            # Wait before retrying to prevent hammering the server
            time.sleep(2 ** captcha_retry_attempt)  # Exponential backoff

    if captcha_retry_attempt == MAX_CAPTCHA_RETRIES:
        # If the CAPTCHA still couldn't be found after all retries, return failure
        error_message = "Failed to extract CAPTCHA after multiple attempts."
        print(error_message)
        queue.put({
            "level": "ERROR",
            "message": error_message,
            "timestamp": time.time()
        })
        return None

    # Solve CAPTCHA
    captcha_text = solve_captcha(base64_image, session, queue)
    if not captcha_text:
        return None

    # Load and prepare the form data
    with open(passport_json_file, 'r') as file:
        data = json.load(file)

    # Add config values to the passport data
    data['openingPeriodId'] = config['openingPeriodId']
    data['date'] = config['date']
    data['dateStr'] = config['dateStr']
    data['captchaText'] = captcha_text

    # Log the preparation of form data
    queue.put({
        "level": "DEBUG",
        "message": f"Prepared form data for {passport_json_file} with CAPTCHA.",
        "timestamp": time.time()
    })

    # POST request with retry policy and logging
    show_form_url2 = config['showForm_url_post']
    # wait_process(config['form_post_time'])
    time.sleep(config['delay_form'])  # Simulate delay

    post_response = retry_request(
        session, 'POST', show_form_url2, queue,
        data=data, headers=headers, proxies=proxies, verify=False
    )

    if post_response is None:
        error_message = f"Failed to submit showForm after multiple attempts."
        print(error_message)
        queue.put({
            "level": "INFO",
            "message": error_message,
            "timestamp": time.time()
        })
        return None

    # Check if the POST request was successful
    if post_response.status_code == 200:
        success_message = "Form data submitted successfully."
        print(success_message)
        queue.put({
            "level": "INFO",
            "message": success_message,
            "timestamp": time.time()
        })
        return post_response

    # Handle non-200 status codes
    error_message = f"Failed to submit showForm. Status code: {post_response.status_code}"
    print(error_message)
    queue.put({
        "level": "INFO",
        "message": error_message,
        "timestamp": time.time()
    })

    return None




# Function to process a single passport
def process_single_passport(passport_json_file, queue, post_time=None, get_time=None):
    # Load captcha model and wait for time
    wait_until_specific_time(post_time, queue)

    """Process a single passport and log messages to the queue."""
    try:
        session = requests.Session(impersonate='chrome124')
        first_captcha_url = (
            "https://service2.diplo.de/rktermin/extern/appointment_showMonth.do?locationCode=kara&realmId=771&categoryId=1416&dateStr=23.01.2025"
        )
        date_post_file = 'date_post.json'

        # Log debug message
        queue.put({
            "level": "DEBUG",
            "message": f"Starting CAPTCHA process for {passport_json_file},  attempt {1}",
            "timestamp": time.time()
        })

        session, response = first_captcha_solver(session, first_captcha_url, date_post_file, queue, post_time)

        if response:
            # Use the URL from config_mine.json
            show_form_url = config['showForm_url']
            response = process_show_form(session, show_form_url, passport_json_file, queue, get_time)

            if response:
                success_message = f"Form successfully submitted for {passport_json_file}: {response.url}"
                print(success_message)

                # Log success message
                queue.put({
                    "level": "SUCCESS",
                    "message": success_message,
                    "timestamp": time.time()
                })
                if "thanx" in response.url:
                    success_message = f"Appointment successfully reserved for {passport_json_file}"
                    print(success_message)

                    # Log success message
                    queue.put({
                        "level": "SUCCESS",
                        "message": success_message,
                        "timestamp": time.time()
                    })
            else:
                error_message = f"Form submission failed for {passport_json_file}"
                print(error_message)

                # Log info message for failure
                queue.put({
                    "level": "INFO",
                    "message": error_message,
                    "timestamp": time.time()
                })
        else:
            error_message = f"Failed to process first CAPTCHA for {passport_json_file}"
            print(error_message)

            # Log info message for CAPTCHA failure
            queue.put({
                "level": "INFO",
                "message": error_message,
                "timestamp": time.time()
            })

    except Exception as e:
        error_message = f"An error occurred for {passport_json_file}: {str(e)}"
        print(error_message)

        # Log error message
        queue.put({
            "level": "INFO",
            "message": error_message,
            "timestamp": time.time()
        })



def run_multiprocessing(post_time=None, get_time=None):
    passport_dir = 'passports/'
    passport_files = [os.path.join(passport_dir, f) for f in os.listdir(passport_dir) if f.endswith('.json')]

    process_target_num = math.ceil(mp.cpu_count()*config['num_process_per_core']) - 1
    # process_target_num = 1
    # passport_target_num = min(config['num_passports'], process_target_num)
    passport_target_files = random.choices(passport_files, k=process_target_num)
    print(f"Starting with {len(passport_target_files)} processes on system with {mp.cpu_count()} cores for passports: {passport_target_files}",)


    with mp.Manager() as manager:
        queue = manager.Queue()
        print(f"Initial queue size: {queue.qsize()}")

        log_process = mp.Process(target=log_sender, args=(queue,))
        log_process.start()

        num_processes = len(passport_target_files)

        queue.put({
            "level": "INFO",
            "message": f"Using {random_config}, starting with {num_processes} processes on system with {mp.cpu_count()} cores for passports: {passport_target_files}",
            "timestamp": time.time()
        })

        try:
            with Pool(num_processes) as pool:
                pool.starmap(
                    process_single_passport,
                    [(passport_file, queue, post_time, get_time) for passport_file in passport_target_files]
                )
        finally:
            # Ensure the sentinel value is always sent
            queue.put(None)
            log_process.join(timeout=10)  # Wait for up to 10 seconds
            if log_process.is_alive():
                print("Log process did not terminate, forcibly terminating")
                log_process.terminate()
                log_process.join()

        print(f"Final queue size: {queue.qsize()}")


# Main function
if __name__ == "__main__":
    post_time = config['post_time']  # Koha e caktuar për POST-in nga config_mine.json
    get_time = config['get_time']  # Koha e caktuar për GET-in nga config_mine.json

    run_multiprocessing(post_time, get_time)
