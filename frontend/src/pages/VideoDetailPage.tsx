import { useNavigate, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getVideoVideosVideoIdGet, listJobsByVideoVideosVideoIdJobsGet } from "../api/generated";
import { authHeaders } from "../lib/auth";

export default function VideoDetailPage() {
  const navigate = useNavigate();
  const { id } = useParams();

  const { data, isLoading } = useQuery({
    queryKey: ["video", id],
    queryFn: () => getVideoVideosVideoIdGet(id!, { headers: authHeaders() }),
  });

  const { data: jobsData } = useQuery({
    queryKey: ["jobs", id],
    queryFn: () => listJobsByVideoVideosVideoIdJobsGet(id!, { headers: authHeaders() }),
  });
    
  return (
    <div>
      {isLoading ? (
        <p>読み込み中...</p>
      ) : (
        <div>
          {data?.status === 200 && (
            <div>
              <h1 className="text-2xl font-bold">{data.data.title}</h1>
              <p>{new Date(data.data.created_at).toLocaleDateString("ja-JP")}</p>
              <p>{data.data.status}</p>
            </div>
          )}
        </div>
      )}

      {jobsData?.status === 200 && (
        <ul>
          {jobsData.data.map((job) => (
            <li key={job.id}>{job.status}</li>
          ))}
        </ul>
      )}

      <button
        onClick={() => navigate("/")}
        className="mt-4 rounded bg-blue-500 px-4 py-2 text-white hover:bg-blue-600"
      >
        一覧に戻る
      </button>
    </div>
  );
}