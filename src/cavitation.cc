//----------------------------------------------------------------------
#include <vector>
#include <assert.h>
#include <fstream>
#include <algorithm>
#include "mpistream.h"
#include "confmaker.h"
#include "communicator.h"
#include "cavitation.h"
//----------------------------------------------------------------------
Cavitation cav;
const double density_threshold = 0.2;
//----------------------------------------------------------------------
class BubbleHist {
private:
  double grid_size;
  int local_grid_x;
  int local_grid_y;
  int local_grid_z;
  int local_grid_number;
  int global_grid_number;
  int global_grid_x;
  int global_grid_y;
  int global_grid_z;
  std::vector < std::vector<int> > v_index;
  std::vector < std::vector<unsigned char> > v_data;
  std::vector <int> global_index;
  std::vector <unsigned char> global_data;
  std::vector <unsigned char> global_data_tmp;
  /*
    std::vector <bool> clustering_grid;
    std::vector <int> cluster_number;
    std::vector <int> cluster_size;
  */
  unsigned char threshold;
  int analyse_count;

public:
  BubbleHist(MDManager *mdm, const double gsize) {
    analyse_count = 0;
    MDRect * myrect = mdm->GetMDUnit(0)->GetRect();
    const double wx = myrect->GetWidth(X);
    const double wy = myrect->GetWidth(Y);
    const double wz = myrect->GetWidth(Z);
    local_grid_x = static_cast<int>(wx / gsize + 0.5);
    local_grid_y = static_cast<int>(wy / gsize + 0.5);
    local_grid_z = static_cast<int>(wz / gsize + 0.5);

    int g[D];
    mdm->GetGridSize(g);
    global_grid_x = local_grid_x * g[X];
    global_grid_y = local_grid_y * g[Y];
    global_grid_z = local_grid_z * g[Z];

    grid_size = wx / static_cast<double>(local_grid_x);
    threshold = static_cast<unsigned char>(grid_size * grid_size * grid_size * density_threshold);
    local_grid_number = local_grid_x * local_grid_y * local_grid_z;
    const int num_threads = mdm->GetTotalThreads();
    const int num_units = mdm->GetTotalUnits();
    v_data.resize(num_threads);
    v_index.resize(num_threads);
    if (0 == mdm->GetRank()) {
      global_grid_number = local_grid_number * num_units;
      global_data.resize(global_grid_number);
      global_data_tmp.resize(global_grid_number);
      global_index.resize(global_grid_number);
      /*
        clustering_grid.resize(global_grid_number);
        cluster_number.resize(global_grid_number);
        cluster_size.resize(global_grid_number);
      */
    }
    for (int i = 0; i < num_threads; i++) {
      v_data[i].resize(local_grid_number);
      v_index[i].resize(local_grid_number);
    }
    mout << "# Grid Information ------" << std::endl;
    mout << "# grid_size = " << grid_size << std::endl;
    mout << "# grid: " << global_grid_x << "," << global_grid_y << "," << global_grid_z  << std::endl;
    mout << "# grid_count = " << global_grid_number << std::endl;
    mout << "# ------" << std::endl;

  };

  void SaveDensity(void) {
    char filename[256];
    sprintf(filename, "conf%04d.dat", analyse_count);
    std::ofstream ofs(filename);
    ofs.write((char*)&global_data[0], sizeof(global_data[0]) * (global_data.size()));
  };

  /*
    void SaveDistribution(void) {
      char filename[256];
      std::vector <int> v_dist;
      v_dist.resize(global_grid_number);
      for (int i = 0; i < global_grid_number; i++) {
        v_dist[i] = 0;
      }
      for (int i = 0; i < global_grid_number; i++) {
        v_dist[cluster_size[i]]++;
      }
      sprintf(filename, "d%04d.dist", analyse_count);
      std::ofstream ofs(filename);
      const double v = grid_size * grid_size * grid_size;
      for (int i = 1; i <= cluster_size[0]; i++) {
        if (v_dist[i] != 0) {
          ofs << v * i << " " << v_dist[i] << std::endl;
        }
      }
    };
  */
  void AnalyseSub(MDUnit *mdu, const int index) {
    double *s = mdu->GetRect()->GetStartPosition();
    const int sx = static_cast<int>(s[X] / grid_size + 0.5);
    const int sy = static_cast<int>(s[Y] / grid_size + 0.5);
    const int sz = static_cast<int>(s[Z] / grid_size + 0.5);
    for (int i = 0; i < local_grid_number; i++) {
      const int ix = (i % local_grid_x) + sx;
      const int iy = ((i / local_grid_x) % local_grid_y) + sy;
      const int iz = i / local_grid_x / local_grid_y + sz;
      const int j = ix + iy * global_grid_x + iz * global_grid_x * global_grid_y;
      v_index[index][i] = j;
    }
    for (int i = 0; i < local_grid_number; i++) {
      v_data[index][i] = 0;
    }
    Variables *vars = mdu->GetVariables();
    const int pn = vars->GetParticleNumber();
    double (*q)[D] = vars->q;
    const double gsinv = 1.0 / grid_size;
    for (int i = 0; i < pn; i++) {
      const int ix = static_cast<int>((q[i][X] - s[X]) * gsinv);
      const int iy = static_cast<int>((q[i][Y] - s[Y]) * gsinv);
      const int iz = static_cast<int>((q[i][Z] - s[Z]) * gsinv);
      const int local_index = ix + iy * local_grid_x + iz * local_grid_x * local_grid_y;
      assert(local_index >= 0);
      assert(local_index < local_grid_number);
      v_data[index][local_index] ++;
    }
  };
  void Analyse(MDManager *mdm) {
    mdm->MakePairList();
    const int num_threads = mdm->GetTotalThreads();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_threads; i++) {
      AnalyseSub(mdm->GetMDUnit(i), i);
    }
    std::vector<int> v_indextmp;
    std::vector<unsigned char> v_datatmp;
    for (int i = 0; i < num_threads; i++) {
      v_indextmp.insert(v_indextmp.end(), v_index[i].begin(), v_index[i].end());
      v_datatmp.insert(v_datatmp.end(), v_data[i].begin(), v_data[i].end());
    }
    const int num_procs = mdm->GetTotalProcs();
    Communicator::GatherIntegerVector(v_indextmp, global_index, num_procs, 0);
    Communicator::GatherUCharVector(v_datatmp, global_data_tmp, num_procs, 0);
    if (0 == mdm->GetRank()) {
      for (int i = 0; i < (int)global_index.size(); i++) {
        global_data[global_index[i]] = global_data_tmp[i];
      }
      SaveDensity();
      //Clastering();
      //SaveDistribution();
    }
    analyse_count++;
  };
  int GetIndex(int ix, int iy, int iz) {
    if (ix < 0) {
      ix += global_grid_x;
    } else if (ix >= global_grid_x) {
      ix -= global_grid_x;
    }
    if (iy < 0) {
      iy += global_grid_y;
    } else if (iy >= global_grid_y) {
      iy -= global_grid_y;
    }
    if (iz < 0) {
      iz += global_grid_z;
    } else if (iz >= global_grid_z) {
      iz -= global_grid_z;
    }
    return ix + iy * global_grid_x + iz * global_grid_x * global_grid_y;
  };
  /*
    int GetClusterIndex(int index) {
      while (index != cluster_number[index]) {
        index = cluster_number[index];
      }
      return index;
    };
    void Check(int ix1, int iy1, int iz1, int ix2, int iy2, int iz2) {
      const int i1 = GetIndex(ix1, iy1, iz1);
      const int i2 = GetIndex(ix2, iy2, iz2);
      assert(i1 < global_grid_number);
      assert(i2 < global_grid_number);
      if (!clustering_grid[i1] || !clustering_grid[i2]) {
        return;
      }
      int c1 = GetClusterIndex(i1);
      int c2 = GetClusterIndex(i2);
      if (c1 < c2) {
        cluster_number[c2] = c1;
      } else {
        cluster_number[c1] = c2;
      }
    };
    void Clastering(void) {
      for (int i = 0; i < global_grid_number; i++) {
        clustering_grid[i] = (global_data[i] <= threshold);
        cluster_number[i] = i;
        cluster_size[i] = 0;
      }
      for (int iz = 0; iz < global_grid_z; iz++) {
        for (int iy = 0; iy < global_grid_y; iy++) {
          for (int ix = 0; ix < global_grid_x; ix++) {
            Check(ix, iy, iz, ix + 1, iy  , iz);
            Check(ix, iy, iz, ix  , iy + 1, iz);
            Check(ix, iy, iz, ix  , iy  , iz + 1);
          }
        }
      }
      for (int i = 0; i < global_grid_number; i++) {
        const int index = GetClusterIndex(i);
        cluster_size[index]++;
      }
      std::sort(cluster_size.begin(), cluster_size.end(), std::greater<int>());
    };
    double GetMaxClusterSize(MDManager *mdm) {
      if (mdm->GetRank() != 0) return 0.0;
      const double v = grid_size * grid_size * grid_size;
      return cluster_size[0] * v;
    };
  */
};
//----------------------------------------------------------------------
void
Cavitation::Run(MDManager *mdm) {
  Parameter *param = mdm->GetParameter();
  ExtractConfigurationMaker c(param);
  mdm->ExecuteAll(&c);
  const double v0 = param->GetDoubleDef("InitialVelocity", 1.0);
  mdm->SetInitialVelocity(v0);
  const int T_LOOP = param->GetIntegerDef("ThermalizeLoop", 150);
  const int LOOP = param->GetIntegerDef("TotalLoop", 1000);
  const int OBSERVE_LOOP = param->GetIntegerDef("ObserveLoop", 100);
  mdm->SetControlTemperature(true);
  mdm->ShowSystemInformation();
  double t1 = MPI_Wtime();
  for (int i = 0; i < T_LOOP; i++) {
    mdm->Calculate();
    if (i % OBSERVE_LOOP == 0) {
      mout << mdm->GetSimulationTime();
      mout << " " << mdm->Temperature();
      mout << " " << mdm->PotentialEnergy();
      mout << " " << mdm->Pressure();
      mout << " " << mdm->TotalEnergy();
      mout << " " << MPI_Wtime() - t1;
      mout << " # Thermalize" << std::endl;
    }
  }
  const double CHANGE_SCALE = param->GetDoubleDef("ChangeScale", 1.05);
  const double GRID_SIZE = param->GetDoubleDef("GridSize", 3.0);
  mdm->ChangeScale(CHANGE_SCALE);
  mdm->SetControlTemperature(false);
  BubbleHist bhist(mdm, GRID_SIZE);
  mdm->ShowSystemInformation();
  for (int i = 0; i <= LOOP; i++) {
    mdm->Calculate();
    if (i % OBSERVE_LOOP == 0) {
      bhist.Analyse(mdm);
      mout << mdm->GetSimulationTime();
      mout << " " << mdm->Temperature();
      mout << " " << mdm->PotentialEnergy();
      mout << " " << mdm->Pressure();
      mout << " " << mdm->TotalEnergy();
      mout << " " << MPI_Wtime() - t1;
      mout << " # Observe" << std::endl;
    }
  }
  const std::string filename = param->GetStringDef("OutputFile", "std.out");
  mout.SaveToFile(filename.c_str());
}
//----------------------------------------------------------------------
