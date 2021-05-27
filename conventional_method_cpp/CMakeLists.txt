cmake_minimum_required(VERSION 3.17.0)

project(CONVENTIONAL_EVALUATION)
set(CMAKE_BUILD_PARALLEL_LEVEL)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# opencv
file(TO_CMAKE_PATH ../../3rdParty/OpenCV_4.4.0/ opencv_path)

set(OpenCV_DIR ${opencv_path})
find_package(OpenCV COMPONENTS core highgui imgproc videoio videostab)
include_directories(
	${OpenCV_INCLUDE_DIRS}
 )

add_executable(${PROJECT_NAME} 
	main.cpp
	)


target_link_libraries(${PROJECT_NAME} PRIVATE
	${OpenCV_LIBS})

#OpenCV
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_core440d.dll>$<$<CONFIG:release>:opencv_core440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_highgui440d.dll>$<$<CONFIG:release>:opencv_highgui440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_dnn440d.dll>$<$<CONFIG:release>:opencv_dnn440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_imgproc440d.dll>$<$<CONFIG:release>:opencv_imgproc440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_imgcodecs440d.dll>$<$<CONFIG:release>:opencv_imgcodecs440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_videoio440d.dll>$<$<CONFIG:release>:opencv_videoio440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_videostab440d.dll>$<$<CONFIG:release>:opencv_videostab440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_video440d.dll>$<$<CONFIG:release>:opencv_video440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_calib3d440d.dll>$<$<CONFIG:release>:opencv_calib3d440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_features2d440d.dll>$<$<CONFIG:release>:opencv_features2d440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)
add_custom_command(TARGET CONVENTIONAL_EVALUATION POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different 
	${opencv_path}//x64//vc16//bin//$<$<CONFIG:debug>:opencv_flann440d.dll>$<$<CONFIG:release>:opencv_flann440.dll>
	$<TARGET_FILE_DIR:CONVENTIONAL_EVALUATION>)

