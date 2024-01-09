#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <string.h>


#include <sys/times.h>
#include <cmath>

using namespace std;

#define pop(stack) stack[--stack ## _fill_pointer]
#define push(item, stack) stack[stack ## _fill_pointer++] = item

tms start, finish;
int start_time;

struct Edge{
	int v1;
	int v2;
};

long long	max_steps;
int			cutoff_time;
long long	step;
int			optimal_size;

/*parameters of the instance*/
int		v_num;//|V|: 1...v
int		e_num;//|E|: 0...e-1

/*structures about edge*/
Edge*	edge;

/*structures about vertex*/
int*	dscore;						//dscore of v
//long long	time_stamp[MAXV];
long long* time_stamp;


//from vertex to it's edges and neighbors
int**	v_edges;		//edges related to v, v_edges[i][k] means vertex v_i's k_th edge
int**	v_adj;			//v_adj[v_i][k] = v_j(actually, that is v_i's k_th neighbor)
int*	v_degree;		//amount of edges (neighbors) related to v


/* structures about solution */
//current candidate solution
int		c_size;						//cardinality of C
bool*	v_in_c;						//a flag indicates whether a vertex is in C
int*	remove_cand;				//remove candidates, an array consists of only vertices in C, not including tabu_remove
int*	index_in_remove_cand;
int		remove_cand_size;

//best solution found
//int		best_c_size;
bool*	best_v_in_c;				//a flag indicates whether a vertex is in best solution
double  best_comp_time;
//long    best_step;


//uncovered edge stack
int*	uncov_stack;				//store the uncov edge number
int		uncov_stack_fill_pointer;
int*	index_in_uncov_stack;		//which position is an edge in the uncov_stack


/* functions declaration */
int build_instance(char *filename);
//void init_sol(char *filename);
int init_sol(char *filename);
int check_neighbor(int v);
void remove(int v);



int build_instance(char *filename)
{
	char line[1024];
	char tempstr1[10];
	char tempstr2[10];
	int  v,e;
	
	char	tmp;
	int		v1,v2;

	ifstream infile(filename);
    if(infile==NULL) return 0;

	/*** build problem data structures of the instance ***/
	infile.getline(line,1024);
	while (line[0] != 'p') infile.getline(line,1024);// the first char of the line is not p, meaning that it is an edge (e 344 1)
	sscanf(line, "%s %s %d %d", tempstr1, tempstr2, &v_num, &e_num);// get the first line's information (p edge 516 1188   )
	

	edge = new Edge [e_num];							//be initialized here
	uncov_stack = new int [e_num];                      //only need to initialized uncov_stack_fill_pointer, has been done in init_sol()
	index_in_uncov_stack = new int [e_num];             //the same as above
	dscore = new int [v_num + 1];                       //be initialized in init_sol()
	time_stamp = new long long [v_num + 1];             //be initialized in init_sol()
	v_edges = new int* [v_num + 1];                     //be initialized here
	v_adj = new int* [v_num + 1];                       //the same as above
	v_degree = new int [v_num + 1];                     //the same as above
	memset(v_degree, 0, sizeof(int) * (v_num + 1));     
	v_in_c = new bool [v_num + 1];                      //be initialized in init_sol()
	index_in_remove_cand = new int [v_num + 1];         //the same as above
	best_v_in_c = new bool [v_num + 1];                 //be initialized in update_best_sol() in init_sol()

	for (e=0; e<e_num; e++)
	{
		infile>>tmp>>v1>>v2;
		v_degree[v1]++;
		v_degree[v2]++;
		
		edge[e].v1 = v1;
		edge[e].v2 = v2;
	}
	infile.close();

	for (v=1; v<=v_num; v++)
	{
		v_adj[v] = new int[v_degree[v]];
		v_edges[v] = new int[v_degree[v]];
	}

	
//	int v_degree_tmp[MAXV];
	int* v_degree_tmp = new int [v_num + 1];
	memset(v_degree_tmp, 0, sizeof(int) * (v_num + 1));
	for (e=0; e<e_num; e++)
	{
		v1=edge[e].v1;
		v2=edge[e].v2;

		v_edges[v1][v_degree_tmp[v1]] = e;
		v_edges[v2][v_degree_tmp[v2]] = e;

		v_adj[v1][v_degree_tmp[v1]] = v2;
		v_adj[v2][v_degree_tmp[v2]] = v1;

		v_degree_tmp[v1]++;
		v_degree_tmp[v2]++;
	}
	delete[] v_degree_tmp;
	
	return 1;
}


void free_memory()
{
	for (int v=1; v<=v_num; v++)
	{
		delete[] v_adj[v];
		delete[] v_edges[v];
	}
	delete[] best_v_in_c;
	delete[] index_in_remove_cand;
	delete[] remove_cand;
	delete[] v_in_c;
	delete[] v_degree;
	delete[] v_adj;
	delete[] v_edges;
	delete[] time_stamp;
	delete[] dscore;
	delete[] index_in_uncov_stack;
	delete[] uncov_stack;
	delete[] edge;
}

inline
void uncover(int e) 
{
	index_in_uncov_stack[e] = uncov_stack_fill_pointer;
	push(e,uncov_stack);
}


int check_neighbor(int v)
{
	int i,n;
	int edge_count = v_degree[v];
	for (i=0; i<edge_count; ++i)
	{
		n = v_adj[v][i];
		if (v_in_c[n]==0)
		{
            return 0;
		}

	}
	return 1;
}


int init_sol(char *filename)
{
	int v,e;
	int v1, v2;
    int if_cover;
    char line[1024];
    char tmp;

    ifstream infile(filename);
    if(infile==NULL) return 0;
	
	memset(dscore, 0, sizeof(int) * (v_num + 1));
	memset(time_stamp, 0, sizeof(long long) * (v_num + 1));
	memset(v_in_c, 0, sizeof(bool) * (v_num + 1));

	c_size = 0;
;
	for (v=1; v<=v_num; v++)
	{
        infile>>tmp>>if_cover;
        if (if_cover==1)
        {
            v_in_c[v] = 1;
            c_size++;
        }
        else v_in_c[v] = 0;
	}
	infile.close();

	for (v=1; v<=v_num; v++)
	{
	if (v_in_c[v]==1)
       if (check_neighbor(v)==1)
            v_in_c[v] = 0;
            c_size--;
	}

    int real_c_size=0;
    for (v=1; v<=v_num; v++)
	{
	if (v_in_c[v]==1)
            real_c_size++;
	}
    c_size = real_c_size;
	times(&finish);
	best_comp_time = double(finish.tms_utime - start.tms_utime + finish.tms_stime - start.tms_stime)/sysconf(_SC_CLK_TCK);
	best_comp_time = round(best_comp_time * 100)/100.0;
	return 1;
}

void remove(int v)
{
	v_in_c[v] = 0;
	dscore[v] = -dscore[v];
	int i,e,n;

	int edge_count = v_degree[v];
	for (i=0; i<edge_count; ++i)
	{
		e = v_edges[v][i];
		n = v_adj[v][i];

		if (v_in_c[n]==0)
		{
			dscore[n]++;
			uncover(e);
		}
		else
		{
			dscore[n]--; 
		}
	}
}

