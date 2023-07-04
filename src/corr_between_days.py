# In this file iam going to be comparing the correlation between the days in terms of spikes found in the data

# Wallboxes:
"""
wallboxes = [
        DataWallboxesColumns.PowerKEBA1,
        DataWallboxesColumns.PowerKEBA2,
        DataWallboxesColumns.PowerKEBA3,
        DataWallboxesColumns.PowerLadebox1,
        DataWallboxesColumns.PowerLadebox2,
        DataWallboxesColumns.PowerLadebox3,
        DataWallboxesColumns.PowerDeltaWallbox,
        DataWallboxesColumns.PowerRaption50,
    ] 
"""
import polars as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Correlation between days

def corr_days(data: pl.DataFrame, window, segment) -> pl.DataFrame:
    # shape of data

    # take only the window of the data
    data = data.slice(window[0], window[1])

    #collect the data for each day of the week
    weekdays = data.groupby("DayOfWeek")


    #creating an empty list to store the data for each day of the week
    days = [None] * 7

    for day, group in weekdays:
        days[day - 1] = group

    # now we have a list of arrays, each array is a day of the week
    mondays, tuesdays, wednesdays, thursdays, fridays, saturdays, sundays = days
    
    #convert the arrays into numpy arrays
    mondays = np.array(mondays)
    tuesdays = np.array(tuesdays)
    wednesdays = np.array(wednesdays)
    thursdays = np.array(thursdays)
    fridays = np.array(fridays)
    saturdays = np.array(saturdays)
    sundays = np.array(sundays)

    days = [mondays, tuesdays, wednesdays, thursdays, fridays, saturdays, sundays]

    # 7 plots for each day of the week with the data for each day

    dict_days = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    def compare_simmilarity_between_days(days, segment):
        # Machine learning approach
        # we are going to use kmeans to cluster the data
        # 7 days of the week, we need to find which days are similar to each other

        # days is a list of arrays, each array is a day of the week

        # days is a list of arrays, each array is a day of the week
        # combine all the days into one array
        days = np.concatenate(days)
        model = KMeans(n_clusters=7)
        model.fit(days)
        model.predict(days)
        model.cluster_centers_
        model.labels_
        model.inertia_

        plt.figure()
        plt.scatter(days[:, 0], days[:, 1], c=model.labels_, cmap="rainbow")
        plt.title(f"Kmeans clustering for segment {segment}")
        plt.xlabel("Time of the day in days")
        plt.ylabel("Power in kW")
        plt.xticks(np.arange(0, len(days), step=1), np.arange(0, 24 * 7, step=1))
        plt.savefig(f"plots/kmeans_segment_{segment}.png")
        plt.show()

    compare_simmilarity_between_days(days, segment=segment)


    # we are going to cluster the data for each day of the week and see if we can find any patterns
    def cluster_parts_of_the_day(counter, day, segment):
        scores = []
        # we are going to use the elbow method to find the optimal number of clusters
        # we are going to use the inertia score
        for i in range(2, 7):
            model = KMeans(n_clusters=i)
            model.fit(day)
            model.predict(day)
            model.cluster_centers_
            scores.append(model.inertia_)

        plt.figure()
        plt.plot(np.arange(2, 7), scores)
        plt.title(f"Elbow method for day {dict_days[counter]}_{segment}")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.show()
    
        #after looking at the elbow method we can see that the optimal number of clusters is 4
        optimal_number_of_clusters = 4
        
        
        #create the model with the optimal number of clusters
        model = KMeans(n_clusters=optimal_number_of_clusters)
        model.fit(day)
        #predict the clusters
        model.predict(day)
        model.cluster_centers_
        model.labels_
        model.inertia_


        #plot the clusters
        plt.figure()
        plt.scatter(day[:, 0], day[:, 1], c=model.labels_, cmap="rainbow")
        plt.title(f"Kmeans clustering for day {dict_days[counter]}_{segment}")
        plt.xlabel("Time of the day in hours")
        plt.ylabel("Power in kW")
        plt.savefig(
            f"plots/part_of_day_clustering{dict_days[counter]}_segment_{segment}.png"
        )
        plt.show()

    counter = 0

    # for day in days:
    #    cluster_parts_of_the_day(counter,day, segment=segment)
    #    counter += 1
    def viz_days():
        counter = 0
        for day in days:
            plt.figure()
            x = np.arange(0, len(day))

            # y is the second column of the data which is the power
            y = day[:, 1]

            plt.plot(x, y)
            plt.title(f"Battery day of the week {dict_days[counter]} segment {segment}")

            plt.xlabel("Time of the day in hours")
            plt.ylabel("Power in kW")
            # on the x axis we have the time of the day in minutes but we want to have it in hours
            # so we divide by 60
            # and only show the hours
            try:
                plt.xticks(np.arange(0, len(day), step=60), np.arange(0, 24, step=1))
                print("day", dict_days[counter])
                print("minimum", np.min(y))
                print("maximum", np.max(y))
                print(" ")

            except:
                pass
                # save the plot as a png in a format
            # plt.savefig(f"plots/{segment}/battery_day_of_the_week_{dict_days[counter]}_segment_{segment}.png")
            counter += 1
            # plt.show()

    # viz_days()
    # visualize days together in one plot
    """ for seg in range(1,8):
        plt.figure()
        for day in days:
            x = np.arange(0, len(day))
            y = day[:, 1]
            plt.plot(x, y)
            #vuew the plot only between 8 and 14 hours
            
            plt.title(f"Battery segment {seg}")
            plt.xlabel("Time of the day in hours")
            plt.ylabel("Power in kW")
            plt.legend(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            try:
                plt.xticks(np.arange(0, len(day), step=60), np.arange(0, 24, step=1))
            except:
                pass
        
        plt.xlim(8*60, 14*60)
        plt.ylim(-7, 0)
        plt.savefig(f"plots/united{seg}.png")
        #plt.show() """
