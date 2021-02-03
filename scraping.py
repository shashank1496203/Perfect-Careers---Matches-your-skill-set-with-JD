from selenium import webdriver
import pandas as pd
from bs4 import BeautifulSoup
import requests
from itertools import zip_longest
import time
import os
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def checkCaptcha(webDriver):
    try:
        webDriver.find_element_by_xpath("//div[not(@id='invisible-recaptcha-div') and @class='g-recaptcha']")
        print("Detected Captcha")
        time.sleep(120)
        return True
    except:
        return False

def launchNewDriver():
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--incognito")
    driver = webdriver.Chrome(executable_path='/Users/shivanideo/Documents/WebMiningProject/chromedriver', options=chrome_options)
    return driver

title = []
summary = []
totalJobCount = 0

positions = ['Software Engineer','Data Engineer', 'Data Scientist']
posLength = 3
Cities = ['Jersey City, NJ', 'Santa Clara, CA', 'Herndon, VA', 'Seattle, WA', 'San Jose, CA', 'Austin, TX', 'Oakland, CA']
citiesLength = len(Cities)

driver = launchNewDriver()
driver.get("https://www.indeed.com/jobs?q&l=Jersey%20City%2C%20NJ&vjk=c4567b22aa14fe9d")
checkCaptcha(driver)

time.sleep(5)
for job in positions:
    jobCount = 0
    pathName = job +"_HTML_Files"
    if(not os.path.exists(pathName)):
        os.mkdir(pathName)  #Create a HTML folder for current job
    for city in Cities:
        try: 
            jobField = driver.find_element_by_id("what") #Find job field text box
            jobField.clear()
            jobField.send_keys(job)
            locationField = driver.find_element_by_id("where") #Find location field text box
            locationField.clear()
            locationField.send_keys(city)
            time.sleep(5)
            #driver.find_element_by_class_name("jobsearch-Autocomplete-list-container").click() #Click on dropdown
            time.sleep(5)
            driver.find_element_by_class_name("input_submit").click()
            time.sleep(5)
            noOfJobsInCity =  int(driver.find_element_by_id("searchCountPages").text.split(" ")[3].replace(',',''))
            
            if(noOfJobsInCity >= 4000):
                noOfJobsInCity = noOfJobsInCity * 0.40 #If city has more than 4000 jobs, get 40% of those
            else:
                noOfJobsInCity = noOfJobsInCity * 0.75 #If city has less than 4000 jobs, get 50% of those
            
            currentCount = 0
            job_links = []
            checkCaptcha(driver)       
            time.sleep(4)

            while(currentCount < noOfJobsInCity):
                currentLink = driver.current_url#Save current URL for nagivation later
                print(currentLink)
                checkCaptcha(driver)
                time.sleep(5)
                summaryItems = driver.find_elements_by_css_selector("a.jobtitle.turnstileLink")
                job_links = [summaryItem.get_attribute("href") for summaryItem in summaryItems]
                for job_link in job_links:
                    jobCount += 1
                    currentCount +=1
                    totalJobCount += 1

                    driver.get(job_link)
                    time.sleep(5)
                    checkCaptcha(driver)
                    time.sleep(5)
            
                    try:
                        job_title = driver.find_element_by_css_selector("h1.icl-u-xs-mb--xs.icl-u-xs-mt--none.jobsearch-JobInfoHeader-title").text
                        title.append(job_title)
                    except:
                        job_title='None'
                    try:
                        job_description = driver.find_element_by_css_selector("div.jobsearch-jobDescriptionText").text
                        summary.append(job_description)
                    except:
                        job_description='None'

                    htmlContent = requests.get(job_link).text #Parse HTML of each job link
                    soup = BeautifulSoup(htmlContent, 'html.parser')
                    fileName = "file_"+str(jobCount)
                    f = open(os.path.join(pathName, '%s.html'%(fileName)), 'w',encoding = 'utf-8')
                    f.write(str(soup.prettify())) #Create html file for job
                    f.close()
                job_links.clear()

                driver.get(currentLink) #Navigate to job search page we were on previously
                time.sleep(5)
                checkCaptcha(driver)
                time.sleep(5)
           
                nextElement =  driver.find_element_by_xpath("//li/a[@aria-label='Next']")
                nextElement.click()
                time.sleep(5)
                nextUrl = driver.current_url
                checkCaptcha(driver)

        except:
            final = []
            for item in zip_longest(title,summary):
                final.append(item)
            df4=pd.DataFrame(final,columns=['Job_title','Summary'])
            df4.to_csv("%s.csv" %(job),index=False) 
            break 
            
        if(jobCount >= 6000 or city==Cities[citiesLength-1]): #If job count> 6000 or we have reached end of cities, create csv file
            final = []
            for item in zip_longest(title,summary):
                final.append(item)
            df4=pd.DataFrame(final,columns=['Job_title','Summary'])
            df4.to_csv("%s.csv" %(job),index=False) 
            break

driver.close()

