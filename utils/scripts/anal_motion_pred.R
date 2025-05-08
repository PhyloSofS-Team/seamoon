library("tidyverse")
library("RColorBrewer")
library(reshape2, ggplot2)

# process raw outputs from SeaMoon, which are individual losses between each pair of ground-truth 
# and predicted motions, and identify the best-matching pairs, with the constraint that 
# a prediction cannot be matched to several ground-truth motions
getRes<-function(path,k=1,mintype="row",fnam="list_test.txt",correct=FALSE,boggous=FALSE){

	queries=read.table(fnam)$V1
	res = c()
	prot = c()
	gt = c()
	err = c()
	pred = c()
	for(q in queries){
		if(correct){
			fnam = paste0(path,"/",q,"_losses.csv",sep="")
			#fnam = paste0(path,"/",q,"_corrected.csv",sep="")
		}
		else{
			fnam = paste0(path,"/",q,".csv",sep="")
		}
		# get the corrected losses, PRED are the rows, GT are the columns
		dat = read.table(fnam,head=TRUE,sep=",")
		if(k==1){
			res = rbind(res, dat[1,2:(1+k)])
		}
		else{
			if(mintype=="col"){
				# take the min over each column, meaning for each ground-truth motion
				res = rbind(res, apply(dat[1:k,2:(1+k)],2,min))
			}
			else{
				if(boggous){
					subdat = dat[1:k,1:k]
				}
				else{
					# extract submatrix
					subdat = dat[1:k,2:4]
				}
				# identify the best-matching ground-truth (GT) motion for each prediction (PRED)
				minInd = apply(subdat,1,which.min)
				# get the corresponding min losses
				minVal = apply(subdat,1,min)
				# we now have a vector of minimal losses, of length k
				# it may happen that more than 1 PRED targets the same GT
				# in that case, several losses in this vector will correspond to the same GT
				# and they will thus have the same minInd
				# here, we retain only the minimal loss among them 
				minVal_final=c()
				pind = c()
				gtind = sort(unique(minInd))
				for(ind in gtind){
					minVal_final=c(minVal_final,min(minVal[minInd==ind]))
					pind=c(pind,seq(1,k)[minInd==ind][which.min(minVal[minInd==ind])])
				}
				err = c(err, minVal_final)
				gt = c(gt, gtind)
				pred = c(pred, pind)
				prot = c(prot, rep(q,length(minVal_final)))
			}
		}
	}
	if(mintype=="row"){
		res = data.frame(prot, pred, gt, err) 
	}
	else{
		res = data.frame(Name=queries,res)
		colnames(res) = c("Name","correct_gt1","correct_gt2","correct_gt3")
	}
	return(res)
}

# compute RMSIP between the ground-truth and predicted subspaces
getRMSIP<-function(path,fnam="list_test.txt"){

	queries=read.table(fnam)$V1
	rmsip = c()
	prot = c()
	for(q in queries){
		# get the scalar products, PRED are the rows, GT are the columns
		dat = as.matrix(read.table(paste0(path,"/",q,"/",q,"_dot_mat.txt",sep="")))
		rmsip = c(rmsip, sqrt(sum(dat^2)/dim(dat)[[1]]))
		prot = c(prot, q)
	}
	res = data.frame(prot, rmsip) 
}

# read cosine similarities between all pairs of PRED and GT motions
# and order the values from the best to the worst
getCorrMax<-function(path,fnam="list_test.txt"){

	queries=read.table(fnam)$V1
	res = c()
	prot = c()
	for(q in queries){
		# get the scalar products, PRED are the rows, GT are the columns
		dat = read.table(paste0(path,"/",q,".csv",sep=""),head=TRUE,sep=",")
		dat = dat[,-1]
		tmp = sqrt(apply(dat^2,2,max))
		res = rbind(res, sort(tmp,decreasing=TRUE))
		prot = c(prot, q)
	}
	rownames(res) = prot
	colnames(res) = c("best","mid","worst")
	return(res) 
}


# compute the cummulative sum 
computeCumSum<-function(vals,breaks=seq(0,1,by=0.01),better="low"){

	res = c()
	for(b in breaks){
		if(better=="low"){
			res = c(res, sum(vals<=b))
		}
		else{
			res = c(res, sum(vals>=b))
		}
	}
	#return(cbind(breaks,res/length(vals))) -- for percentages
	return(cbind(breaks,res)) # for counts

}

# compute collectivity of a motion (kappa)
computeColl<-function(mode){

	# normalise the vector
	norm2 = apply(mode,1,f<-function(x){sum(x*x)})
 	norm2 = norm2/sum(norm2)
 	print(c(sum(norm2),length(norm2)))
 	# we sum only over the non-zero norms to avoid NaNs from log(0)
 	numerator = exp(-sum(norm2[norm2>0]*log(norm2[norm2>0])))
	coll = numerator/length(norm2)
	return(list(coll,numerator,norm2))

}


# perform paired Wilcoxon ranked signed test between pairs of methods (reduced version)
computePval<-function(){

	methods = c("esm_test_3v3_1ref","esm_test_3v3_5ref","t5_test_3v3_1ref","t5_test_3v3_5ref","nolb_test")
	n = length(methods)
	print(c("there are",n,"methods"))
	pvalMat = matrix(nc=n,nr=n)
	colnames(pvalMat) = methods
	rownames(pvalMat) = methods
	preds = list()

	for(i in 1:n){
		method = methods[i]
		print(method)
		dat = read.table(paste0(method,".csv"),sep=",",head=TRUE)
		dat_reduced = dat%>%group_by(prot)%>%filter(err==min(err))
		print(dim(dat_reduced))
		preds[[i]] = unlist(dat_reduced[,"err"])
	}
	print(sapply(preds,length))
	for(i in 1:n){
		for(j in 1:n){
			if(i!=j){
				print(c(methods[i],methods[j]))
				pvalMat[i,j]= wilcox.test(preds[[i]],preds[[j]],alter="greater",paired=TRUE)$p.value
			}
		}
	}
	return(pvalMat)
}

# plot the cummulative SSE curves for the CATH categories
plotCumLoss_all_cath<-function(cutoff=0.6){

	ath_train_topol_stats=read.table("cath_train_topol_stats.csv",head=TRUE,sep=";",colClass=c("character","numeric"))
	N = dim(cath_train_topol_stats)[[1]]
	cath_test_all=read.table("cath_test_all.csv",head=TRUE,sep=";",colClass="character")
	# get the folds that appear in the test set
	topol_test = unique(cath_test_all[,"topology"])
	# create a filter for the training set
	fac_topol_test = tapply(cath_train_topol_stats[,"topology"],1:N,f<-function(x){x%in%topol_test})
	cath_train_topol_stats = cath_train_topol_stats[fac_topol_test,]
	# total number of annotated domains in train that also appear in test 
	# (each one has a CATH fold assigned to it)
	nTot = sum(cath_train_topol_stats[,"nb"])
	# compute the proportion of each fold in the training set
	cath_train_topol_stats[,"nb"] = cath_train_topol_stats[,"nb"] / nTot
	# get the folds assigned to the test proteins
	cath_test_all_folds = merge(cath_test_all,cath_train_topol_stats,by.x="topology",by.y="topology")

	methods = c("esm_test_3v3_1ref","esm_test_3v3_5ref","esm3_test_3v3_1ref","esm3_test_3v3_5ref","t5_test_3v3_1ref","t5_test_3v3_5ref","nolb_test")
	meth_names = c("SeaMoon-ESM2","SeaMoon-ESM2(x5)","SeaMoon-ESM3","SeaMoon-ESM3(x5)","SeaMoon-ProstT5","SeaMoon-ProstT5(x5)","Normal Mode Analysis (NOLB)")
	ltyp = c(rep(c("longdash","solid"),3),"solid")
	lwd = c(rep(c(1,1.5),3),1.5)
	print(c("there are",length(methods),"methods"))
	b = seq(0,cutoff,by=0.05)
	vals = c()
	ltypes = c()
	lwidths = c()

	meth = c()
	x = c()
	cutvals = c(0.2,0.4,0.6)
	successrateTab = matrix(nc=2*length(cutvals),nr=length(methods))
	rownames(successrateTab) = methods
	for(i in 1:length(methods)){
		method = methods[i]
		print(method)
		# read the result table and reduce
		dat_reduced = filter_dat_wColl(method,1)
		dat = merge(dat_reduced,cath_test_all_folds,by.x="prot",by.y="prot")
		nb_top_in_test = tapply(dat[,"superfamily"],dat[,"topology"],length)
		mu_err = tapply(dat[,"err"],dat[,"topology"],mean)
		print(summary(mu_err))
		print(length(mu_err))
		nb_top = tapply(dat[,"nb"],dat[,"topology"],unique)

		# compute success rate for different levels of NSSE
		tmp = c()
		for(mycut in cutvals){
			tmp = c(tmp, sum(mu_err<mycut),round(sum(mu_err<mycut)/length(mu_err)[[1]],dig=2))
		}
		successrateTab[method,] = tmp
		vals = c(vals, computeCumSum(mu_err,breaks=b)[,2])
		meth = c(meth, rep(meth_names[i],length(b)))
		ltypes = c(ltypes, rep(ltyp[i],length(b)))
		lwidths = c(lwidths, rep(lwd[i],length(b)))
		x = c(x,b)
	}
	print(successrateTab)

	vals = vals/length(mu_err)*100
	dat = data.frame(x,vals,meth,ltypes,lwidths)
	dat$meth = factor(dat$meth, levels = c("SeaMoon-ESM2","SeaMoon-ESM2(x5)","SeaMoon-ESM3","SeaMoon-ESM3(x5)","SeaMoon-ProstT5","SeaMoon-ProstT5(x5)","Normal Mode Analysis (NOLB)"))
	p = ggplot(dat,aes(y=vals,x=x,group=meth,color=meth)) + geom_line(linetype=ltypes,size=1)
	p = p + theme_light() + ylim(c(0,max(vals)))
	p = p+ theme(legend.text=element_text(size=12),axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"))
	p = p + ylab("Percentage of test proteins") + xlab("Normalised sum of squares error")
	myCol = c("lightblue3","royalblue","pink","magenta","gold","orange","firebrick")
	p = p + scale_color_manual(values=myCol)
	p = p + theme(legend.position="none")#legend.title=element_blank())

	# assess statistical significance of performance differences
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5(x5)"],dat$vals[dat$meth=="SeaMoon-ESM3(x5)"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5"],dat$vals[dat$meth=="SeaMoon-ESM3"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5(x5)"],dat$vals[dat$meth=="SeaMoon-ESM2(x5)"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5"],dat$vals[dat$meth=="SeaMoon-ESM2"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ESM3(x5)"],dat$vals[dat$meth=="SeaMoon-ESM3"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5(x5)"],dat$vals[dat$meth=="Normal Mode Analysis (NOLB)"],paired=TRUE,alter="greater"))
	plot(c(min(b),max(b)),c(0,max(res)),col="white",xlab = "Sum of squared error",ylab="Number of test proteins",cex.axis=1.3,cex.lab=1.5)

	return(p)
}

# plot the cummulative SSE curves
plotCumLoss_all<-function(setup,cutoff=0.6,seltm=1.2,selid=1.2){

	methods = c("esm_test_3v3_1ref","esm_test_3v3_5ref","esm3_test_3v3_1ref","esm3_test_3v3_5ref","t5_test_3v3_1ref","t5_test_3v3_5ref","nolb_test")
	meth_names = c("SeaMoon-ESM2","SeaMoon-ESM2(x5)","SeaMoon-ESM3","SeaMoon-ESM3(x5)","SeaMoon-ProstT5","SeaMoon-ProstT5(x5)","Normal Mode Analysis (NOLB)")
	ltyp = c(rep(c("longdash","solid"),3),"solid")
	lwd = c(rep(c(1,1.5),3),1.5)
	print(c("there are",length(methods),"methods"))
	b = seq(0,cutoff,by=0.02)
	vals = c()
	ltypes = c()
	lwidths = c()

	meth = c()
	x = c()
	cutvals = c(0.2,0.4,0.6)
	successrateTab = matrix(nc=2*length(cutvals),nr=length(methods))
	rownames(successrateTab) = methods
	for(i in 1:length(methods)){
		method = methods[i]
		print(method)
		# read the result table and reduce
		dat_reduced = filter_dat_wKpax(method,1.2) #filter_dat(method,1.2)
		if(setup==1){
			dat_reduced = dat_reduced[dat_reduced[,"id"]>selid,]
		}
		else{
			if(setup==2){
				dat_reduced = dat_reduced[dat_reduced[,"id"]<selid,]
				#dat_reduced = dat_reduced[dat_reduced[,"tm"]>seltm,]
				dat_reduced = dat_reduced[dat_reduced[,"kpax.TM.Score.Flex"]>seltm,]
			}
			else{
				dat_reduced = dat_reduced[dat_reduced[,"id"]<selid,]
				#dat_reduced = dat_reduced[dat_reduced[,"tm"]<seltm,]
				dat_reduced = dat_reduced[dat_reduced[,"kpax.TM.Score.Flex"]<seltm,]
			}
		}
		# compute success rate for different levels of NSSE
		tmp = c()
		for(mycut in cutvals){
			tmp = c(tmp, sum(dat_reduced[,"err"]<mycut),round(sum(dat_reduced[,"err"]<mycut)/dim(dat_reduced)[[1]],dig=2))
		}
		successrateTab[method,] = tmp
		vals = c(vals, computeCumSum(dat_reduced[,"err"],breaks=b)[,2])
		meth = c(meth, rep(meth_names[i],length(b)))
		ltypes = c(ltypes, rep(ltyp[i],length(b)))
		lwidths = c(lwidths, rep(lwd[i],length(b)))
		x = c(x,b)
	}
	print(successrateTab)

	vals = vals/dim(dat_reduced)[[1]]*100
	dat = data.frame(x,vals,meth,ltypes,lwidths)
	dat$meth = factor(dat$meth, levels = c("SeaMoon-ESM2","SeaMoon-ESM2(x5)","SeaMoon-ESM3","SeaMoon-ESM3(x5)","SeaMoon-ProstT5","SeaMoon-ProstT5(x5)","Normal Mode Analysis (NOLB)"))
	p = ggplot(dat,aes(y=vals,x=x,group=meth,color=meth)) + geom_line(linetype=ltypes,size=1)
	p = p + theme_light() + ylim(c(0,max(vals)))
	p = p+ theme(legend.text=element_text(size=10),axis.text=element_text(size=12),axis.title=element_text(size=14,face="bold"))
	p = p + ylab("Percentage of test proteins") + xlab("Normalised sum of squares error") + labs(color = "") 
	myCol = c("lightblue3","royalblue","pink","magenta","gold","orange","firebrick")
	p = p + scale_color_manual(values=myCol) 
	p = p + theme(legend.position=c(0.3,0.75))#"none")#legend.title=element_blank())

	# assess statistical significance of performance differences
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5(x5)"],dat$vals[dat$meth=="SeaMoon-ESM3(x5)"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5"],dat$vals[dat$meth=="SeaMoon-ESM3"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5(x5)"],dat$vals[dat$meth=="SeaMoon-ESM2(x5)"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ProstT5"],dat$vals[dat$meth=="SeaMoon-ESM2"],paired=TRUE,alter="greater"))
	print(t.test(dat$vals[dat$meth=="SeaMoon-ESM3(x5)"],dat$vals[dat$meth=="SeaMoon-ESM3"],paired=TRUE,alter="greater"))
#	plot(c(min(b),max(b)),c(0,max(res)),col="white",xlab = "Sum of squared error",ylab="Number of test proteins",cex.axis=1.3,cex.lab=1.5)
#	for(i in 1:length(methods)){
#		lines(b,res[,i],lwd=3,col=myCol[i])
#	}
#	legend(0,max(res),c("SeaMoon-ESM2","SeaMoon-ESM2(x5)","SeaMoon-ProstT5","SeaMoon-ProstT5(x5)","Normal Mode Analysis (NOLB)"),fill=myCol,border="white",bty="n",cex=1.3)
	return(p)
}


# inset for Figure showing performance (Fig. 2)
# for each pair (X,Y) of the selected methods,
# take the top 100 best predicted proteins from method X
# and compute how many have a certain quality range with method Y
plotBarsIntercept<-function(n_test=1121){
	
	#methods = c("esm_test_3v3_5ref","esm3_test_3v3_5ref","t5_test_3v3_5ref","nolb_test")
	methods = c("esm_test_3v3_5ref","t5_test_3v3_5ref","nolb_test")
	print(c("there are",length(methods),"methods"))
	combi = list(c(1,3),c(3,1),c(2,3),c(3,2))
	#combi = list(c(4,1),c(4,2),c(4,3))
	thresh = c(0.60,0.75,1.1)

	# get the top 100 for all methods
	top100 = list()
	dat_reduced = list()
	for(i in 1:length(methods)){
		method = methods[i]
		print(method)
		dat = read.table(paste0(method,".csv"),sep=",",head=TRUE)
		dat = dat[order(dat[,"err"]),]
		# reduce by taking the min error per protein
		dat_reduced[[i]] = dat%>%group_by(prot)%>%filter(err==min(err))
		# retain the first 100 proteins
		top100[[i]] = unlist(dat_reduced[[i]][1:100,"prot"])
		print(top100[[i]][90:100])
	}
	
	vals = c()
	combis = c()
	threshs = c()
	# for each combination of methods
	for(c in combi){
		print(c)
		tmp = c()
		for(t in thresh){
			i = c[1]
			j = c[2]
			# among all test proteins, take those that have a certain quality for method Y
			goodj = unlist(dat_reduced[[j]][which(dat_reduced[[j]][,"err"]<=t),"prot"])
			# intersect with the top 100 from method X
			if(length(tmp)==0){
				tmp=c(tmp,length(intersect(top100[[i]],goodj)))
			}
			else{
				mySum = sum(tmp)
				tmp=c(tmp,length(intersect(top100[[i]],goodj))-mySum)
			}
		}
		threshs = c(threshs, c("Acceptable","Intermediate","Inaccurate"))
		vals = c(vals,tmp)
		combis = c(combis,rep(toString(c),length(thresh)))
	}
	res = data.frame(combis,threshs,vals)
	print(res)
	res$threshs <- factor(res$threshs, levels = rev(c("Acceptable","Intermediate","Inaccurate")))

	p = ggplot(res,aes(y=vals,x=combis,fill=threshs)) + geom_bar(stat="identity")
	p = p + theme_light() 
	p = p+ theme(legend.text=element_text(size=14),axis.text=element_text(size=22),axis.title=element_text(size=20,face="bold"))
	#p = p + xlab("")+ ylab("Number of proteins") + labs(col="Sum of squares error")
	p = p + xlab("")+ ylab("")
	p = p + scale_fill_manual(values=rev(c(gray(0.2),gray(0.8),gray(0.4))))
	#p = p + theme(legend.title=element_blank())

	return(p)

}


# get the reuslts from file
# for a given method and filter based on the error
filter_dat_wKpax<-function(method,cutoff=0.6){

	pred = read.table(paste0(method,".csv"),sep=",",head=TRUE)
	pred_reduced = pred%>%group_by(prot)%>%filter(err==min(err))
	stats_test = read.table("stats_test.csv",sep=",",head=TRUE)
	dat= merge(pred_reduced,stats_test,by.x="prot",by.y="name_file")
	kpax = read.table("struct_sim_max_train_test_kpax.csv",head=TRUE,sep=",")
	dat2 = merge(dat,kpax,by.x="prot",by.y="id_ligne_test")
	#print(dat%>%filter(over>0.7))

	dat2 = dat2%>%filter(err<=cutoff)
	dat2 = dat2[order(dat2[,"err"]),]
	return(dat2)

}

# get the reuslts from file
# for a given method and filter based on the error
filter_dat<-function(method,cutoff=0.6){

	pred = read.table(paste0(method,".csv"),sep=",",head=TRUE)
	pred_reduced = pred%>%group_by(prot)%>%filter(err==min(err))
	stats_test = read.table("stats_test.csv",sep=",",head=TRUE)
	dat= merge(pred_reduced,stats_test,by.x="prot",by.y="name_file")
	#print(dat%>%filter(over>0.7))

	dat = dat%>%filter(err<=cutoff)
	dat = dat[order(dat[,"err"]),]
	return(dat)

}

# performs an hypergeometric test
hgt<-function(hitInSample,hitInPop,sampleSize,popSize){
  reshyper=NULL
  failInPop=popSize-hitInPop
  if(sampleSize>0){
    if(hitInSample/sampleSize>hitInPop/popSize){reshyper= phyper(hitInSample-1, hitInPop, failInPop, sampleSize, lower.tail= FALSE)}
    else{reshyper=-phyper(hitInSample, hitInPop, failInPop, sampleSize, lower.tail= TRUE)}}
  else{reshyper=1}
  return(reshyper)
}

# transform a p-value into a log-value
logpvalue<-function(pval){
  if(pval>0) logpval=-log10(pval)
  else logpval=log10(-pval)
  return(logpval)
}

# test for enrichment or depletion in different categories of collectivity
getEnrich_coll<-function(cutoff=0.6){

	methods = c("esm_test_3v3_5ref","esm3_test_3v3_5ref","t5_test_3v3_5ref","nolb_test")
	meth_names = c("ESM2\n(x5)","ESM3\n(x5)","ProstT5\n(x5)","NMA")
	coll_test = as.matrix(read.table("coll_test.csv",head=TRUE,sep=";"))
	coll_test_cat = coll_test
	coll_test_cat[coll_test<=0.3] = "local"
	coll_test_cat[coll_test>0.3&coll_test<=0.6] = "regional"
	coll_test_cat[coll_test>0.6] = "global"
	hitInPop =  table(coll_test_cat)
	hitInPop = hitInPop[c(2,3,1)]
	popSize = length(coll_test_cat)
	print(popSize)
	print(round(hitInPop/popSize,dig=2))
	myCat = c()
	myMet = c()
	myLogFoldChange = c()
	myPval = c()
	for(k in 1:length(methods)){
		method = methods[k]
		pred = read.table(paste0(method,".csv"),sep=",",head=TRUE)
		kappa = c()
		for(i in 1:dim(pred)[[1]]){
			kappa = c(kappa, coll_test[pred[i,"prot"],pred[i,"gt"]])
		}
		coll_class = rep("local",dim(pred)[[1]])
		coll_class[kappa>0.6] = "global"
		coll_class[kappa>0.30&kappa<=0.6] = "regional"
		pred = data.frame(pred, kappa, size=coll_class)
		pred = pred[pred[,"err"]<cutoff,]
		hitInSample = table(pred[,"size"])
		hitInSample = hitInSample[c(2,3,1)]
		sampleSize = dim(pred)[[1]]
		print(method)
		print(sampleSize)
		print(round(hitInSample/sampleSize,dig=2))
		for (i in 1:3){
			myMet = c(myMet, meth_names[k])
			myCat = c(myCat, names(hitInSample)[i])
			myLogFoldChange = c(myLogFoldChange, log2(hitInSample[i]*popSize/hitInPop[i]/sampleSize))
			myPval = c(myPval, hgt(hitInSample[i],hitInPop[i],sampleSize,popSize))
		}
	}
	dat = data.frame(method=myMet,size=myCat, LFC=myLogFoldChange, pval=myPval)
	dat$method = factor(dat$method,levels=meth_names)
	dat$size = factor(dat$size, levels = names(hitInSample))
	print(dat)
	p = ggplot(dat, aes(x=method,y=LFC,fill=size)) + geom_bar(position="dodge",stat="identity")
	p = p + theme_light() 
	p = p+ theme(legend.text=element_text(size=14),axis.text=element_text(size=16),legend.title=element_text(size=18,face="bold"),axis.title=element_text(size=18,face="bold"))
	p = p + xlab("Method")+ ylab("Log fold change") + labs(fill="Motion\nsize")
	p = p + theme(legend.position=c(0.2, 0.2))
	p = p + scale_fill_viridis_d(begin=0,end=0.8,direction=-1)

	return(p)

}

# perform bootstrap simulations for stability analysis
compute_success_rate<-function(N=1000){

	methods = c("esm_test_3v3_5ref","esm3_test_3v3_5ref","t5_test_3v3_5ref","nolb_test")
	meth_names = c("SeaMoon-ESM2(x5)","SeaMoon-ESM3(x5)","SeaMoon-ProstT5(x5)","Normal Mode Analysis (NOLB)")

	M = length(meth_names)
	thresh = c(0.2,0.4,0.6)
	trueSR = c()
	mu = c()
	sigma = c()
	CI = c()
	lowlim50 = c()
	uplim50 = c()
	lowlim95 = c()
	uplim95 = c()
	meth = c()
	threshold = c()
	for(i in 1:M){
		method = methods[i]
		pred = filter_dat(method,1.2)
		print(dim(pred))
		n = dim(pred)[[1]]
		m = n^(2/3)#sqrt(n)
		for(t in thresh){
			SR = c()
			for(k in 1:N){
				err = sample(pred[,"err"], m, replace = TRUE)
				SR = c(SR, sum(err<t)/m)
			}
			SR = sort(SR)
			if(t==0.6){hist(SR)}
			mu = c(mu, mean(SR))
			trueSR = c(trueSR, sum(pred[,"err"]<t)/n)
			sigma = c(sigma, sd(SR))
			tmp = 1.96*sd(SR)/sqrt(N)
			CI = c(CI, tmp)
			lowlim95 = c(lowlim95, SR[round((N-1)*0.025)])
			uplim95 = c(uplim95, SR[round((N-1)*0.975)])
			lowlim50 = c(lowlim50, SR[round((N-1)*0.25)])
			uplim50 = c(uplim50, SR[round((N-1)*0.75)])
			#lowlim = c(lowlim, mean(SR)-tmp)
			#uplim = c(uplim, mean(SR)+tmp)
			meth = c(meth, meth_names[i])
			threshold = c(threshold, paste("<",toString(t)))
		}
	}
	dat = data.frame(SR=trueSR,mu,sigma,CI,lowlim95,uplim95,lowlim50,uplim50,method=meth,threshold)
	dat$method = factor(dat$method, levels = meth_names)
	p = ggplot(dat,aes(y=mu*100,x=threshold,fill=method)) + geom_bar(stat="identity", position='dodge')#,width = 0.8, position = position_dodge(width = 0.8))
	p = p + geom_errorbar(aes(x=threshold, ymin=lowlim95*100, ymax=uplim95*100), lwd=0.5, col="grey", width=0.7, position=position_dodge(width = 0.9))
	p = p + geom_errorbar(aes(x=threshold, ymin=lowlim50*100, ymax=uplim50*100), lwd=0.6, width=0.5, position=position_dodge(width = 0.9))
	p = p + theme_light() 
	p = p+ theme(legend.text=element_text(size=14),axis.text=element_text(size=16),axis.title=element_text(size=20,face="bold"))
	#p = p + theme(axis.text.x=element_text(angle=90,vjust=0.5,hjust=1))
	p = p + ylab("Success rate")+ xlab("Normalised sum-of-squares error") + labs(fill="")
	myCol = c("royalblue","magenta","orange","firebrick")
	p = p + scale_fill_manual(values=myCol)
	p = p + theme(legend.position = c(0.3,0.85))

	return(list(dat,p))
}

# scatterplots for the iMod benchmark
plotiMod<-function(setup=1){

	ch = read.table("imod_bench_info.txt",head=TRUE,sep=";")
	pdb_nam = ch[,"protein"]
	n = length(pdb_nam)
	print(n)

	datColl = read.table("coll_test.csv",head=TRUE,sep=";")
	print(datColl[1,])
	err = c()
	errNMA = c()
	prot = c()
	kappa = c()
	mtype = c()
	seqid = c()
	tm = c()
	for(i in 1:n){
		pred = read.table(paste0("redo_all_results/redo_base_5ref_2024-06-18_22-46-27/",pdb_nam[i],".csv"),head=TRUE,sep=",",row.names=1)
	#	pred = read.table(paste0("redo_all_results/5ref_3v3_ESM_2024-04-04_05-27-58/",pdb_nam[i],".csv"),head=TRUE,sep=",",row.names=1)
		err = c(err, min(pred))
		predNMA = read.table(paste0("nm_mode_loss_linear_HOPMA_mixed_corrected/",pdb_nam[i],"_corrected.csv"),head=TRUE,sep=",",row.names=1)
		errNMA = c(errNMA, min(predNMA[1:3,]))
		#selNMA = which(predNMA[,"prot"]==pdb_nam[i]&predNMA[,"gt"]==1)
		#if(length(selNMA)==0){print(paste("warning!!",pdb_nam[i]))}
		#err = c(err, predNMA[predNMA[,"prot"]==pdb_nam[i]&predNMA[,"gt"]==1,"err"])
		prot = c(prot, rep(strsplit(pdb_nam[i],"_")[[1]][2],1))
		kappa = c(kappa, rep(max(datColl[pdb_nam[i],1:3]),1))
		mtype = c(mtype,ch[i,"motion"])
		seqid = c(seqid,ch[i,"seqid"])
		tm = c(tm,ch[i,"tmflex"])
	}
	
	n = length(err)
	print(n)
	method = rep(c("SeaMoon-ProstT5(x5)","NMA"),n)

	dat = data.frame(prot,kappa,err,errNMA,mtype,seqid,tm)
	print(summary(kappa))
	print(summary(seqid*100))
	print(dat[order(err),])
	
	if(setup==1){
		p = ggplot(dat, aes(x=errNMA, y=err, color=mtype)) 
	}
	else{
		p = ggplot(dat, aes(x=errNMA, y=err, color=seqid*100))
	}
	p = p + geom_rect(aes(xmin = 0, xmax = 0.6, ymin = 0, ymax = 0.6),fill = gray(0.9),colour=NA)
	if(setup==1){
		p = p + geom_point(mapping = aes(size=kappa*5)) + labs(col="Type of motions")  
		p = p + scale_size(name   = "Collectivity",
			 range = c(4*2/3,10*2/3),
             breaks = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1)*5,
             labels = expression(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
		p = p +   guides(
   		# Légende de taille avec cercles vides
    	size = guide_legend(override.aes = list(
      	fill = NA,
      	shape = 21,
      	color = "black"  # Couleur de la bordure des cercles vides
    	)),
    	# Légende de couleur 
    	color = guide_legend(override.aes = list(size=4))
    	)
	}
	else{
		p = p + geom_point(mapping = aes(size=tm*5)) + labs(col="Sequence \nidentity(%)")
		p = p + scale_size(name   = "Flex TM-score",
			 range = c(7*2/3,10*2/3),
             breaks = c(0.2, 0.4, 0.6, 0.7, 0.8,0.9,1)*5,
             labels = expression(0.2, 0.4, 0.6, 0.7, 0.8,0.9,1))
		p = p +   guides(
   		# Légende de taille avec cercles vides
    	size = guide_legend(override.aes = list(
      	fill = NA,
      	shape = 21,
      	color = "black"  # Couleur de la bordure des cercles vides
    	)))
	}
	p = p + geom_hline(aes(yintercept=0.6),lwd=0.8,lty="dashed",col=gray(0.8))
	p = p + geom_vline(aes(xintercept=0.6),lwd=0.8,lty="dashed",col=gray(0.8))
	p = p + theme_light() 
	p = p + scale_x_continuous(limits = c(0, 1), expand = c(0, 0.02)) + scale_y_continuous(limits = c(0, 1), expand = c(0, 0.02))
	p = p+ theme(legend.text=element_text(size=12),legend.title=element_text(size=14),axis.text=element_text(size=14),axis.title=element_text(size=18,face="bold"))
	p = p + xlab("NMA")+ ylab("SeaMoon-ProstT5(x5)") 
	p = p + geom_abline(slope=1,intercept=0, linetype="solid", color = "black")
#	p = p + theme(legend.position=c(0.2,0.8))
	if(setup==2){
		p = p + scale_color_viridis_c(direction=-1)
	}
	return(p)

}

# get the reuslts from file
# for a given method and filter based on the error
filter_dat_wColl<-function(method,cutoff=0.6){

	pred = read.table(paste0(method,".csv"),sep=",",head=TRUE)
	coll_test = read.table("coll_test.csv",head=TRUE,sep=";")
	kappa = c()
	neff = c()
	for(i in 1:dim(pred)[[1]]){
		kappa = c(kappa, coll_test[pred[i,"prot"],pred[i,"gt"]])
		neff = c(neff, neff_test[pred[i,"prot"],pred[i,"gt"]])
	}
	pred = data.frame(pred, kappa, neff)
	pred_reduced = pred%>%group_by(prot)%>%filter(err==min(err))
	stats_test = read.table("stats_test.csv",sep=",",head=TRUE)
	dat= merge(pred_reduced,stats_test,by.x="prot",by.y="name_file")

	dat = dat%>%filter(err<=cutoff)
	dat = dat[order(dat[,"err"]),]
	return(dat)
}

# scatterplot of the error in function of similarity to the train set
plotLoss_vs_sim<-function(method,simStruct="tm",cutoff=0.6){

	dat = filter_dat_wKpax(method,cutoff)
	#print(dat%>%filter(err<=0.2&tm<0.6))  
	n = dim(dat)[[1]]
	
	print(cor(dat[,simStruct],dat[,"err"]))
	print(cor(dat[,"id"],dat[,"err"]))
	print(sum(dat[,simStruct]<0.5))
	print(sum(dat[,"err"]<0.6&dat[,simStruct]<0.5))
	p = ggplot(dat, aes(x=dat[,simStruct],y=err,col=id)) + geom_point(size=2)
	p = p + theme_light() +xlim(c(0,1))
	p = p+ theme(legend.text=element_text(size=14),axis.text=element_text(size=16),axis.title=element_text(size=20,face="bold"))
	p = p + xlab("TM-score")+ ylab("Sum-of-squares error") + labs(col="Sequence\nidentity (%)")
	p = p + scale_color_viridis_c()

	return(p)

}


# compute some statistical summary of the errors per folds
plot_stat_err_fold<-function(method){
	cath_train_topol_stats=read.table("cath_train_topol_stats.csv",head=TRUE,sep=";",colClass=c("character","numeric"))
	N = dim(cath_train_topol_stats)[[1]]
	cath_test_all=read.table("cath_test_all.csv",head=TRUE,sep=";",colClass="character")
	# get the folds that appear in the test set
	topol_test = unique(cath_test_all[,"topology"])
	# create a filter for the training set
	fac_topol_test = tapply(cath_train_topol_stats[,"topology"],1:N,f<-function(x){x%in%topol_test})
	cath_train_topol_stats = cath_train_topol_stats[fac_topol_test,]
	# total number of annotated domains in train that also appear in test 
	# (each one has a CATH fold assigned to it)
	nTot = sum(cath_train_topol_stats[,"nb"])
	# compute the proportion of each fold in the training set
	#cath_train_topol_stats[,"nb"] = cath_train_topol_stats[,"nb"] / nTot
	# get the folds assigned to the test proteins
	cath_test_all_folds = merge(cath_test_all,cath_train_topol_stats,by.x="topology",by.y="topology")
	dat_reduced = filter_dat_wColl(method,1)
	# assign the per-protein errors to the corresponding folds
	dat = merge(dat_reduced,cath_test_all_folds,by.x="prot",by.y="prot")
	#print(dat[1:3,])
	mu_err = tapply(dat[,"err"],dat[,"topology"],mean)
	min_err = tapply(dat[,"err"],dat[,"topology"],min)
	med_err = tapply(dat[,"err"],dat[,"topology"],median)
	nb_top = tapply(dat[,"nb"],dat[,"topology"],unique)
	print(c(length(mu_err),sum(mu_err<=0.6),sum(mu_err<=0.6)/length(mu_err)))
	dat = data.frame(nb_top,mu_err,min_err,med_err)
	print(summary(lm(mu_err~nb_top)))	
	print(summary(lm(min_err~nb_top)))
	p = ggplot(dat,aes(nb_top,min_err)) + geom_point(col=rgb(0,0,1,alpha=0.4),size=2) + scale_x_log10()
	p = p + theme_light() + ylim(c(0,1))
	p = p+ theme(legend.text=element_text(size=12),axis.text=element_text(size=12),axis.title=element_text(size=16,face="bold"))
	p = p + xlab("Number of train proteins")+ ylab("NSSE") 
	return(p)
}

# scatterplot of error in function of collectivity and colored by CATH occurrence
plotLoss_vs_coll_fold<-function(method,cutoff=0.6){

	cath_train_topol_stats=read.table("cath_train_topol_stats.csv",head=TRUE,sep=";",colClass=c("character","numeric"))
	cath_test_all=read.table("cath_test_all.csv",head=TRUE,sep=";",colClass="character")
	cath_test_all_nbtrain = merge(cath_test_all,cath_train_topol_stats,by.x="topology",by.y="topology")
	nb_max = tapply(cath_test_all_nbtrain[,"nb"],cath_test_all_nbtrain[,"prot"],max)
	cath_test_all_nbtrain_max = data.frame(prot=names(nb_max),nbcath=nb_max)

	dat_reduced = filter_dat_wColl(method,cutoff)
	dat = merge(dat_reduced,cath_test_all_nbtrain_max,by.x="prot",by.y="prot")
	#print(dat%>%filter(err<=0.2&tm<0.6))
	n = dim(dat)[[1]]
	print(paste("there are only",n,"dots"))
	
	print(cor(dat[,"kappa"],dat[,"err"]))
	print(cor(log(dat[,"nbcath"]),dat[,"err"]))
	selcath = dat[,"nbcath"]==1
	print(c(sum(dat[,"err"]<0.6&selcath),sum(selcath),sum(dat[,"err"]<0.6&selcath)/sum(selcath)))
	selcath = dat[,"nbcath"]>1&dat[,"nbcath"]<=5
	print(c(sum(dat[,"err"]<0.6&selcath),sum(selcath),sum(dat[,"err"]<0.6&selcath)/sum(selcath)))
	selcath = dat[,"nbcath"]>5&dat[,"nbcath"]<=50
	print(c(sum(dat[,"err"]<0.6&selcath),sum(selcath),sum(dat[,"err"]<0.6&selcath)/sum(selcath)))
	selcath = dat[,"nbcath"]>50&dat[,"nbcath"]<=100	
	print(c(sum(dat[,"err"]<0.6&selcath),sum(selcath),sum(dat[,"err"]<0.6&selcath)/sum(selcath)))
	selcath = dat[,"nbcath"]>50&dat[,"nbcath"]>100	
	print(c(sum(dat[,"err"]<0.6&selcath),sum(selcath),sum(dat[,"err"]<0.6&selcath)/sum(selcath)))
	p = ggplot(dat, aes(x=kappa,y=err,col=nbcath)) + geom_point(size=2)
	p = p + theme_light() 
	p = p+ theme(legend.text=element_text(size=14),axis.text=element_text(size=16),axis.title=element_text(size=20,face="bold"),legend.title=element_text(size=16))
	p = p + xlab("Collectivity")+ ylab("Sum-of-squares error") + labs(col="Fold \ncounts")
	p = p + scale_color_viridis_c(trans = "log10", direction = -1)

	return(p)

}

# plot overlapping densities of errors dor the different methods
plotNMAErrorDensity<-function(methods=c("esm_test_3v3_5ref","t5_test_3v3_5ref"),cutoff=1.1){

	dat1 = filter_dat(methods[1],cutoff)
	dat2 = filter_dat(methods[2],cutoff)[c("prot","err")]
	dat = merge(dat1,dat2,by.x="prot",by.y="prot")
	datNMA = filter_dat("nolb_test",cutoff)[c("prot","err")]
	dat = merge(dat,datNMA,by.x="prot",by.y="prot")

	sel_low=dat[,"err.x"]<0.25&dat[,"err.y"]<0.25
	sel_high=dat[,"err.x"]>0.75&dat[,"err.y"]>0.75

	d_all = density(dat[,c("err")])
	d_low = density(dat[sel_low,c("err")])
	d_high = density(dat[sel_high,c("err")])

	print(c(sum(sel_low),sum(sel_high)))
	print(c(median(dat[sel_low,c("err")]),median(dat[sel_high,c("err")])))

	max_val = max(c(d_all$y,d_low$y,d_high$y))

	plot(d_all,lwd=6,main="",cex.axis=1.3,cex.lab=1.5,xlab="",ylim=c(0,max_val),xaxt="n")
	polygon(d_all, col=rgb(0.7,0.7,0.7), border="black")
	lines(d_low,col="royalblue",lwd=2)
	polygon(d_low, col=rgb(0.6784314,0.8470588,0.9019608,alpha=0.3), border="royalblue")
	lines(d_high,col="tomato",lwd=2)
	polygon(d_high, col=rgb(1.0000000,0.3882353,0.2784314,alpha=0.3), border="tomato")

}

# scatterplot for comparing the errors of 2 methods
plotScatter2Methods<-function(methods,cutoff=0.6){

	dat1 = filter_dat(methods[1],cutoff)
	dat2 = filter_dat(methods[2],cutoff)[c("prot","err")]
	dat = merge(dat1,dat2,by.x="prot",by.y="prot")
	print(cor(dat[,"err.x"],dat[,"err.y"]))
	print(cor(dat[,"err.x"]-dat[,"err.y"],dat[,"nb_members"]))
	datNMA = filter_dat("nolb_test",cutoff)[c("prot","err")]
	dat = merge(dat,datNMA,by.x="prot",by.y="prot")
	print(dat[1,])
	print(dim(dat))
	print(cor(dat[,"err.y"]-dat[,"err.x"],dat[,"err"]))
	dat = dat[order(dat[,"err"],decreasing=TRUE),]
	c=0.6
	print(c(sum(dat[,"err.x"]<c)/dim(dat)[[1]],sum(dat[,"err.x"]<c&dat[,"err"]<c)/sum(dat[,"err"]<c),2*sum(dat[,"err.x"]<c&dat[,"err"]<c)/(sum(dat[,"err"]<c)+sum(dat[,"err.x"]<c))))
	print(c(sum(dat[,"err.y"]<c)/dim(dat)[[1]],sum(dat[,"err.y"]<c&dat[,"err"]<c)/sum(dat[,"err"]<c),2*sum(dat[,"err.y"]<c&dat[,"err"]<c)/(sum(dat[,"err"]<c)+sum(dat[,"err.y"]<c))))

	p = ggplot(dat, aes(x=err.x,y=err.y,col=err)) + geom_point(size=2)
	p = p + theme_light() 
	p = p+ theme(legend.title = element_text(size=18),legend.text=element_text(size=12),axis.text=element_text(size=16),axis.title=element_text(size=20,face="bold"))
	p = p + xlab("SeaMoon-ESM2(x5)")+ ylab("SeaMoon-ProstT5(x5)") + labs(col="NMA")
	p = p + scale_color_viridis_c(alpha=0.9)
	p = p + geom_abline(slope=1,intercept=0, linetype="dashed", color = "black")

	return(p)

}

# plot results for ablation studies
plotAblation<-function(){

	N_prot=rev(c(439,271,411,413,375,402,119,177,0))
	typ = rev(c("baseline",rep("architecture",2),rep("loss",3),rep("input",2),"random"))
	#myCol = c("gold","brown","brown","palegreen","palegreen","palegreen","grey","grey","black")
	myCol = rev(c("gold","brown","palegreen","grey","black"))
	names(myCol) = unique(typ)
	setups=rev(c("baseline","conv kernel=1","transformer","wt. sign flip","wt. permutation","wt. reflection","random embeddings","positional encoding only","random weights"))
	dat = data.frame(setups,N_prot,typ)
	print(dat)
	dat$setups = factor(setups,levels=setups)
	dat$typ = factor(typ,levels=unique(typ))
	p = ggplot(dat,aes(setups,N_prot,fill=typ)) + geom_bar(stat="identity", position = "dodge")
	p = p + coord_flip() + theme_light() 
	p = p+ theme(legend.text=element_text(size=14),axis.text=element_text(size=16),axis.title=element_text(size=20,face="bold"))
	p = p + xlab("")+ ylab("Number of proteins")# + labs(col="Sum of squared error")
	p = p + scale_fill_manual(values=myCol)
	p = p + theme(legend.title=element_blank()) + guides(fill = guide_legend(reverse=T))
	return(p)

}
