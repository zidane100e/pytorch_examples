from datetime import date, timedelta, datetime
import holidays as kr_hol
import collections
import copy as cp

WDAY2STR = {0: 'mon', 1: 'tue', 2: 'wed', 3: 'thu', 4: 'fri', 5: 'sat', 6: 'sun'}
STR2WDAY = {'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 7}

## ADD election date later
PRESIDENT_END_DATES = [date(2022, 5, 9)]
MAYOR_END_DATES = [date(2022, 6, 30)]
LAWMAKER_END_DATES = [date(2024, 5, 29)]

def get_holidays(years):
    if isinstance(years, collections.Sequence) is False:
        years = [years]

    holidays = {}
    for year1 in years:
        holiday1 = Holidays_a_year(year1).kr_holidays
        for key, val in holiday1.items():
            holidays[key] = val
    return holidays

class Holidays_a_year():
    def __init__(self, year=None, flag_holidays=True):
        """
        legal rules change frequently
        replace_rules is made at 2022-04-26. But before this it was different
        flag_holidays is correct at time 2022, but if it is changed you can edit replace_rules by yourself
        """
        if year is None:
            year = date.today().year
        self.kr_holidays = kr_hol.KR(years=year)
        if flag_holidays is True:
            self.dates = list(self.kr_holidays.keys())
        if flag_holidays is False:
            self.kr_holidays = {key: val for key, val in self.kr_holidays.items() if 'Alternative' not in val }
            self.dates = list(self.kr_holidays.keys())
            self.replace()

    def replace(self):
        self.kr_holidaysb = cp.copy(self.kr_holidays)
        for date1, name in self.kr_holidays.items():
            date2 = self.replace_rules(date1, name)
            if date2 != date1:
                self.kr_holidaysb[date2] = 'replace ' + name
        self.kr_holidays = self.kr_holidaysb
        self.dates = list(self.kr_holidays.keys())

    def next_busday(self, date1, wday1, flag_seolchu=False):
        prohibit_wday = [5,6]
        if flag_seolchu is True:
            prohibit_wday = [6]
        
        ix_date1 = self.dates.index(date1)
        if ix_date1 == len(self.dates)-1:
            datesb = self.dates[:ix_date1]
        else:
            datesb = self.dates[:ix_date1] + self.dates[ix_date1+1:]
        while date1 in datesb or wday1 in prohibit_wday:
            date1 = date1 + timedelta(days=1)
            wday1 = (wday1 + 1)%7
        return date1
                    
    def replace_rules(self, date1, name):
        wday = datetime.weekday(date1)
        if "Lunar New Year's Day" in name or 'Chuseok' in name:
            return self.next_busday(date1, wday, True)
        elif name in ['Independence Movement Day', "Children's Day", 'Liberation Day', 'National Foundation Day', 'Hangeul Day']:
            return self.next_busday(date1, wday, False)
        else:
            return date1


if __name__ == '__main__':

    holidays = get_holidays(2022)
    for key, val in holidays.items():
        print(key, val)

    print()
    holidays = get_holidays([2020,2021,2022])
    for key, val in holidays.items():
        print(key, val)

