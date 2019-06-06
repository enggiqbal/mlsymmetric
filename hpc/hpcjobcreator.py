import pandas as pd
#singularity exec --nv /extra/hossain/gdocker.simg python3 multi_application.py @@@index@@@ 4 > multi/st_out/output_@@@index@@@.txt

commandString="singularity exec --nv /extra/hossain/gdocker.simg python3 multi_application2.py {0} 2 > outputs/{1}/st_out/output_{0}.txt"

expprefix="binaryvnonsym"

f=open("pbs_template.pbs")
pbs=f.read()

runtext=""

for i in range(1, 12):
    job_name="graphsymbinaryvnonsym"+str(i)
    pbs_job=open("pbs/"+job_name+".pbs", "w")
    pbs_tmp=pbs.replace("@@@job_name@@@", job_name )
    pbs_tmp=pbs_tmp.replace("@@@index@@@",str(i) )
    cs=commandString.format(str(i), expprefix)	
    pbs_tmp=pbs_tmp.replace('@@@commandString@@@',cs )
    pbs_job.write(pbs_tmp)
    pbs_job.close()
    runtext=runtext+ " qsub pbs/" + job_name+".pbs \n"

r=open("hpc_jobs_run.txt", "w")
r.write(runtext)
r.close()
