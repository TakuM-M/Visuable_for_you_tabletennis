import { useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { getVideoVideosVideoIdGet } from "../api/generated";
import { authHeaders } from "../lib/auth";

export default function VideoDetailPage() {
  const { id } = useParams();

  const { data, isLoading } = useQuery({
    queryKey: ["video", id],
    queryFn: () => getVideoVideosVideoIdGet(id!, { headers: authHeaders() }),
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
            </div>
          )}
        </div>
      )}
    </div>
  );
}