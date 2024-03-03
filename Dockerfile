FROM kuokyong1997/zeroshot:latest AS build

FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04

COPY --from=build /opt/conda /opt/conda
ENV PATH="${PATH}:/opt/conda/bin"
