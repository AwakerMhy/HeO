#include "vertex_remove.h"


int main(int argc, char* argv[])
{
	int seed,i;
	if(build_instance(argv[1])!=1){
		cout<<"can't open instance file"<<endl;
		return -1;
	}

	times(&start);
	start_time = start.tms_utime + start.tms_stime;

    if(	init_sol(argv[2])!=1){
    cout<<"can't open result file of HeO"<<endl;
    return -1;
	}
	times(&finish);
	best_comp_time = double(finish.tms_utime - start.tms_utime + finish.tms_stime - start.tms_stime)/sysconf(_SC_CLK_TCK);
	best_comp_time = round(best_comp_time * 100)/100.0;

    cout<<"Final vertex cover size = "<<c_size<<endl;
    cout<<"Time cost = "<<best_comp_time<<endl;

	free_memory();

	return 0;
}
