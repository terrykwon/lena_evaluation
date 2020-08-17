library(Hmisc)
library(ggplot2)
library(readxl)
library(lmSupport)

#Read in data
setwd("/LOCATION OF EXCEL FILE/")
d<-read_xlsx("data/clip_data.xlsx",sheet="LENA_Supplementary")
d2<-d[!d$Singing=="yes",]

#theme for graphs
theme1<-theme(
  strip.text.x = element_text(size = 15, face="bold"),
  plot.background = element_rect(fill="transparent",color="white"),
  plot.title=element_text(hjust=0.5,size=20),
  ,panel.grid.major = element_line(color="gray83")
  ,panel.grid.minor = element_blank()
  ,panel.border = element_rect(fill="transparent"),panel.background=element_rect(fill="white"),strip.background = element_blank(),
  axis.text.x = element_text(size=15),
  axis.text.y = element_text(size=15),  
  axis.title.x = element_text(size=13.5),
  axis.title.y = element_text(size=13.5),
  legend.text=element_text(size=12),
  legend.title=element_text(size=13.5),
  legend.position = "right",)



####Miss, False Alarm, Confusion, IER####
#overall
sum(d$`False Alarm`)/sum(d$Total_speech)
sum(d$Miss)/sum(d$Total_speech)
sum(d$Confusion)/sum(d$Total_speech)
sum(d$`False Alarm`,d$Miss,d$Confusion)/sum(d$Total_speech)
#by clip
varDescribe(d,Digits=3)
#by age
varDescribeBy(d$IER,d$Age2)
t.test(d$IER~d$Age2)
#by number of speakers
varDescribe(d[d$Speakers2=="2",]$IER,Digits=3)
varDescribe(d[d$Speakers2=="3",]$IER,Digits=3)
varDescribe(d[d$Speakers2=="4+",]$IER,Digits=3)
anova(lm(IER~Speakers2,d))
#Test pitch measures
d$Mean_F0<-as.numeric(d$Mean_F0)
d$Duration_female<-as.numeric(d$Duration_female)
d$Mean_F0C <- d$Mean_F0-mean(d$Mean_F0,na.rm=T)
d$Duration_femaleC<-(d$Duration_female-mean(d$Duration_female,na.rm=T))/1000
d$AgeC<-d$Age-12
summary(lm(IER~Mean_F0C+Duration_femaleC+AgeC+Duration_noise+Duration_TV,data=d))

#####AWC####
#Correlations
cor.test(d$AWC_LENA,d$AWC_Human_space,method="pearson")
cor.test(d$AWC_LENA,d$AWC_Human_morpheme,method="pearson")
cor.test(d$AWC_Human_space,d$AWC_Human_morpheme,method="pearson")
cor.test(d$Speech_duration,d$AWC_LENA,method="pearson")
#by age
cor.test(d[d$Age2=="old",]$AWC_LENA,d[d$Age2=="old",]$AWC_Human_space,method="pearson")
cor.test(d[d$Age2=="old",]$AWC_LENA,d[d$Age2=="old",]$AWC_Human_morpheme,method="pearson")
cor.test(d[d$Age2=="young",]$AWC_LENA,d[d$Age2=="young",]$AWC_Human_space,method="pearson")
cor.test(d[d$Age2=="young",]$AWC_LENA,d[d$Age2=="young",]$AWC_Human_morpheme,method="pearson")
#correction for Korean
lm(AWC_Human_space~AWC_LENA,d)
lm(AWC_Human_morpheme~AWC_LENA,d)
#Graphs
ggplot(d,aes(x=AWC_LENA,y=AWC_Human_space))+geom_point()+ggtitle("Adult Word Count")+xlab("LENA Count")+ylab("Human Count (spaces)")+theme1+
  coord_cartesian(y=c(0,800))+geom_abline(slope=1,intercept=0,linetype="dotted")+geom_smooth(method=lm,color="black")

ggplot(d,aes(x=AWC_LENA,y=AWC_Human_morpheme))+geom_point()+ggtitle("Adult Word Count")+xlab("LENA Count")+ylab("Human Count (morphemes)")+theme1+
  coord_cartesian(y=c(0,1500))+geom_abline(slope=1,intercept=0,linetype="dotted")+geom_smooth(method=lm,color="black")


#####CVC####
##Correlations
cor.test(d$CVC_LENA,d$CVC_Human,method="pearson")
#Excluding clips which include singing
cor.test(d2$CVC_LENA,d2$CVC_Human,method="pearson")
#by age
cor.test(d[d$Age2=="old",]$CVC_LENA,d[d$Age2=="old",]$CVC_Human,method="pearson")
cor.test(d[d$Age2=="young",]$CVC_LENA,d[d$Age2=="young",]$CVC_Human,method="pearson")
#Graph
ggplot(d,aes(x=CVC_LENA,y=CVC_Human))+geom_point()+ggtitle("Child Vocalization Count")+xlab("LENA Count")+ylab("Human Count")+theme1+
  coord_cartesian(y=c(0,190),x=c(0,125))+geom_abline(slope=1,intercept=0,linetype="dotted")+geom_smooth(method=lm,color="black")


#####CTC####
##Correlations
cor.test(d$CTC_LENA,d$CTC_Human,method="pearson")
#xcluding clips which include singing
cor.test(d2$CTC_LENA,d2$CTC_Human,method="pearson")
#By age
cor.test(d[d$Age2=="old",]$CTC_LENA,d[d$Age2=="old",]$CTC_Human,method="pearson")
cor.test(d[d$Age2=="young",]$CTC_LENA,d[d$Age2=="young",]$CTC_Human,method="pearson")
#Graph
ggplot(d,aes(x=CTC_LENA,y=CTC_Human))+geom_point()+ggtitle("Conversational Turn Count")+xlab("LENA Count")+ylab("Human Count")+theme1+
  coord_cartesian(y=c(0,90),x=c(0,30))+geom_abline(slope=1,intercept=0,linetype="dotted")+geom_smooth(method=lm,color="black")


