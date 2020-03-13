#include <OpenMesh/Core/IO/MeshIO.hh>

#include <MishMesh/TriMesh.h>
#include <MishMesh/utils.h>
#include <MishMesh/cone_singularities.h>
#include <MishMesh/minimum_spanning_tree.h>

#include <igl/opengl/glfw/Viewer.h>

void visualize_cones_and_mst(igl::opengl::glfw::Viewer &viewer, MishMesh::TriMesh &mesh, std::vector<MishMesh::TriMesh::VertexHandle> &cone_singularities) {
	viewer.data().points.resize(0, 3);
	viewer.data().lines.resize(0, 9);
	auto trees = MishMesh::minimum_spanning_trees(mesh, cone_singularities);

	for(auto vh : cone_singularities) {
		viewer.data().add_points(Eigen::RowVector3d(mesh.point(vh).data()), Eigen::RowVector3d{1, 0, 0});
	}

	std::vector<MishMesh::TriMesh::EdgeHandle> edges;
	for(const auto &tree : trees) {
		edges.insert(edges.end(), tree.begin(), tree.end());
	}
	Eigen::MatrixX3d P1(edges.size(), 3);
	Eigen::MatrixX3d P2(edges.size(), 3);
	for(uint i = 0; i < edges.size(); i++) {
		auto heh = mesh.halfedge_handle(edges[i], 0);
		P1.row(i) = Eigen::RowVector3d(mesh.point(mesh.from_vertex_handle(heh)).data());
		P2.row(i) = Eigen::RowVector3d(mesh.point(mesh.to_vertex_handle(heh)).data());
	}
	viewer.data().add_edges(P1, P2, Eigen::RowVector3d{0, 0, 1});
}

int main(int argc, char *argv[]) {
	MishMesh::TriMesh mesh;
	OpenMesh::IO::read_mesh(mesh, argv[1]);

	Eigen::MatrixX3d V;
	Eigen::MatrixX3i F;
	std::tie(V, F) = MishMesh::convert_to_face_vertex_mesh(mesh);

	igl::opengl::glfw::Viewer viewer;
	viewer.data().set_mesh(V, F);
	viewer.data().set_face_based(true);
	viewer.data().line_width = 2;
	viewer.data().point_size = 10;
	viewer.data().show_lines = false;

	int iterations = 10;

	auto cone_singularities = MishMesh::compute_cone_singularities(mesh, 0.0, 5, MishMesh::ConeAdditionMode::BOTH);
	visualize_cones_and_mst(viewer, mesh, cone_singularities);

	viewer.callback_key_pressed = [&mesh, &cone_singularities](igl::opengl::glfw::Viewer &viewer, uint key, int modifiers)->bool {
		if(key == 45) {
			if(cone_singularities.size() < 2) return false;
			cone_singularities.resize(cone_singularities.size() - 2);
			visualize_cones_and_mst(viewer, mesh, cone_singularities);
		} else if(key == 43) {
			std::set<size_t> cone_set;
			for(auto cone_vh : cone_singularities) {
				cone_set.insert(cone_vh.idx());
			}
			auto new_cone_singularities = MishMesh::compute_cone_singularities(mesh, 0.0, 1, MishMesh::ConeAdditionMode::BOTH, cone_set);
			cone_singularities = new_cone_singularities;
			visualize_cones_and_mst(viewer, mesh, cone_singularities);
		}
		return false;
	};

	viewer.launch();
}
