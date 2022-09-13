from email import header
from email.quoprimime import header_check
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d')
poland_covid= pd.read_csv("poland_covid.csv", parse_dates=['DATE_REPORTED'],date_parser=dateparse)
norway_covid= pd.read_csv("norway_covid.csv", parse_dates=['DATE_REPORTED'],date_parser=dateparse)
france_covid= pd.read_csv("france_covid.csv", parse_dates=['DATE_REPORTED'],date_parser=dateparse)


poland_covid.index = pd.DatetimeIndex(poland_covid.DATE_REPORTED)
#poland_covid=poland_covid.drop(columns=['DATE_REPORTED'])
#print(poland_covid)
#print (poland_covid.dtypes)

#fig, ax = plt.subplots()
#plt.spines["top"].set_visible(False)    
#ax.spines["bottom"].set_visible(False)    
#ax.spines["right"].set_visible(False)    
#ax.spines["left"].set_visible(False)  

#ax.get_xaxis().tick_bottom()    
#ax.get_yaxis().tick_left() 
#plt.xticks(fontsize=12) 


#Plot cases and deaths for Poland
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(poland_covid['DATE_REPORTED'], poland_covid['NEW_CASES'],'g',label='New cases', color='blue')
#ax.set_title("New cases")
plt.title('Poland: New cases',  fontsize=11)
plt.ylabel('Cases')
plt.subplot(2,1,2)


plt.plot(poland_covid['DATE_REPORTED'], poland_covid['NEW_DEATHS'],'g',label='New deaths', color='red')
plt.title('Poland: New deaths',  fontsize=11)
plt.ylabel('Deaths')
plt.tight_layout()
#plt.show()

#Plot cases and deaths for Norway
plt.figure(2)
plt.subplot(2,1,1)
plt.plot(norway_covid['DATE_REPORTED'], norway_covid['NEW_CASES'],'g',label='New cases', color='blue')
#ax.set_title("New cases")
plt.title('Norway: New cases',  fontsize=11)
plt.ylabel('Cases')
plt.subplot(2,1,2)


plt.plot(norway_covid['DATE_REPORTED'], norway_covid['NEW_DEATHS'],'g',label='New deaths', color='red')
plt.title('Norway: New deaths',  fontsize=11)
plt.ylabel('Deaths')
plt.tight_layout()
#plt.show()

#Plot cases and deaths for France
plt.figure(3)
plt.subplot(2,1,1)
plt.plot(france_covid['DATE_REPORTED'], france_covid['NEW_CASES'],'g',label='New cases', color='blue')
#ax.set_title("New cases")
plt.title('France: New cases',  fontsize=11)
plt.ylabel('Cases')
plt.subplot(2,1,2)


plt.plot(france_covid['DATE_REPORTED'], france_covid['NEW_DEATHS'],'g',label='New deaths', color='red')
plt.title('France: New deaths',  fontsize=11)
plt.ylabel('Deaths')
plt.tight_layout()
#plt.show()

#poland_covid=poland_covid.drop(columns=['DATE_REPORTED'])
#poland_covid["MONTH"] = pd.NaT
poland_covid["MONTH"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).month
poland_covid["DAY_OF_YEAR"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).dayofyear
poland_covid["DAY_OF_MONTH"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).day
poland_covid["DAY_OF_WEEK"] = pd.DatetimeIndex(poland_covid['DATE_REPORTED']).dayofweek

norway_covid["MONTH"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).month
norway_covid["DAY_OF_YEAR"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).dayofyear
norway_covid["DAY_OF_MONTH"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).day
norway_covid["DAY_OF_WEEK"] = pd.DatetimeIndex(norway_covid['DATE_REPORTED']).dayofweek

#print(poland_covid['MONTH'])
#poland_covid.head()
#print(poland_covid)
print("dupa")

plt.figure(4)
sns.set()
p=pd.pivot_table(poland_covid, values='NEW_CASES', index=['MONTH'] , columns=['DAY_OF_MONTH'], aggfunc=np.sum)
print(p)
cmap = sns.color_palette("Blues", as_cmap=True)
ax = sns.heatmap(p, cmap=cmap)
ax.set_title('New cases')
plt.show()
#new = poland_covid[['NEW_CASES','MONTH','DAY_OF_MONTH']]
#new = new.to_numpy()
#print(flights["NEW_CASES"])
#flights2=new
#print(new)
#print (flights2)
#np.poland_covid.iloc(["Month"]), poland_covid.iloc(["Year"]),poland_covid.iloc(["New_cases"])
#ax = sns.heatmap(flights2)
#plt.title("Heatmap Flight Data")
#plt.show()


#VACCINATIONS
#Plot vaccinations
plt.figure(5)
plt.subplot(1,3,1)
plt.plot(poland_covid['DATE_REPORTED'], poland_covid['PEOPLE_VACCINATED'],'g',label='New cases', color='orange')
#ax.set_title("New cases")
plt.title('Poland: Vaccinations',  fontsize=11)
plt.ylabel('People vaccinated')

plt.subplot(1,3,2)

plt.plot(norway_covid['DATE_REPORTED'], norway_covid['PEOPLE_VACCINATED'],'g',label='New deaths', color='orange')
plt.title('Norway: Vaccinations',  fontsize=11)
plt.ylabel('People vaccinated')
plt.tight_layout()
#plt.show()

plt.figure(6)
#plt.subplot(1,3,3)
plt.plot(france_covid['DATE_REPORTED'], france_covid['PEOPLE_VACCINATED'],'g',label='New deaths', color='orange')
plt.title('France: Vaccinations',  fontsize=11)
plt.ylabel('People vaccinated')
plt.tight_layout()
ax = plt.gca()
#ax.set_xlim([xmin, xmax])
ax.set_ylim([1, 53990502])
plt.show()

print(france_covid)