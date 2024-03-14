#include "vertex_remove.h"

//int edge_cand;

int main(int argc, char* argv[])
{
	int seed,i;

	//cout<<"c This is FastMVC, a local search solver for the Minimum Vertex Cover problem."<<endl;
	
	if(build_instance(argv[1])!=1){
		cout<<"can't open instance file"<<endl;
		return -1;
	}
	
	//optimal_size=0;
	//i=2;
		
	//sscanf(argv[i++],"%d",&seed);
	//sscanf(argv[i++],"%d",&cutoff_time);

	
	//srand(seed);
	//cout<<seed<<' ';
	//cout<<"c This is FastVC, solving instnce "<<argv[1]<<endl;
	//cout<<argv[1]<<'\t';
		
	times(&start);
	start_time = start.tms_utime + start.tms_stime;

//   	init_sol(argv[2]);

    if(	init_sol(argv[2])!=1){
    cout<<"can't open result file of HeO"<<endl;
    return -1;
	}

   	//the ending time
	times(&finish);
	best_comp_time = double(finish.tms_utime - start.tms_utime + finish.tms_stime - start.tms_stime)/sysconf(_SC_CLK_TCK);
	best_comp_time = round(best_comp_time * 100)/100.0;
   

    cout<<"Final vertex cover size = "<<c_size<<endl;
    //cout<<"c SearchSteps for best found vertex cover = "<<best_step<<endl;
    cout<<"Time cost = "<<best_comp_time<<endl;


	
	free_memory();

	return 0;
}
